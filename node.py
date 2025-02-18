# 
# This file an adaptation and extension of the p2pfl library (https://pypi.org/project/p2pfl/).
# Refer to the LICENSE file for licensing information.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import logging
import os
from datetime import datetime, timedelta
import textwrap

os.environ['WANDB_SILENT'] = 'true'

# Import the requests module
import requests

logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)

import random
import threading
import time

from lightning.pytorch.loggers import WandbLogger, CSVLogger

from base_node import BaseNode
from communication_protocol import CommunicationProtocol
from config import Config
from pytorch.learning.aggregators.fedavg import FedAvg
from pytorch.learning.exceptions import DecodingParamsError, ModelNotMatchingError
from pytorch.lightninglearner import LightningLearner
from role import Role
from utils.observer import Events, Observer


class Node(BaseNode):
    """
    Class based on a base node that allows **FEDERATED LEARNING**.

    Metrics will be saved under a folder with the name of the node.

    Args:
        model: Model to be learned. Careful, model should be compatible with data and the learner.
        data: Dataset to be used in the learning process. Careful, model should be compatible with data and the learner.
        host (str): Host where the node will be listening.
        port (int): Port where the node will be listening.
        learner (NodeLearner): Learner to be used in the learning process. Default: LightningLearner.
        simulation (bool): If True, the node will be simulated. Default: True.
        encrypt (bool): If True, node will encrypt the communications. Default: False.

    Attributes:
        round (int): Round of the learning process.
        totalrounds (int): Total number of rounds of the learning process.
        learner (NodeLearner): Learner to be used in the learning process.
        aggregator (Aggregator): Aggregator to be used in the learning process.
    """

    #####################
    #     Node Init     #
    #####################

    def __init__(
            self,
            idx,
            experiment_name,
            model,
            data,
            host="127.0.0.1",
            port=None,
            config=Config,
            learner=LightningLearner,
            encrypt=False,
    ):
        # Super init
        BaseNode.__init__(self, experiment_name, host, port, encrypt, config)
        Observer.__init__(self)

        self.idx = idx
        logging.debug("[NODE] My idx is {}".format(self.idx))

        # Import configuration file
        self.config = config

        # Learning
        self.round = None
        self.totalrounds = None
        self.__model_initialized = False
        self.__initial_neighbors = []
        self.__start_thread_lock = threading.Lock()

        # Learner and learner logger
        logging.info("[NODE] Tracking CSV enabled")
        csvlogger = CSVLogger(f"{self.log_dir}", name="metrics", version=f"participant_{self.idx}")
        self.learner = learner(model, data, config=self.config, logger=csvlogger)

        logging.info("[NODE] Role: " + str(self.config.participant["device_args"]["role"]))

        # Aggregator
        if self.config.participant["aggregator_args"]["algorithm"] == "FedAvg":
            self.aggregator = FedAvg(node_name=self.get_name(), config=self.config)

        self.aggregator.add_observer(self)

        self.shared_metrics = False

        # Store the parameters of the model
        self.__stored_model_parameters = []
        self.__timeout = datetime.now()

        # Train Set Votes
        self.__train_set = []
        self.__train_set_votes = {}
        self.__train_set_votes_lock = threading.Lock()

        # Locks
        self.__wait_votes_ready_lock = threading.Lock()
        self.__finish_aggregation_lock = threading.Lock()
        self.__finish_aggregation_lock.acquire()
        self.__wait_init_model_lock = threading.Lock()
        self.__wait_init_model_lock.acquire()
        # Grace period to wait for last transmission using Aggregator thread
        self.__wait_finish_experiment_lock = threading.Lock()
        self.__wait_finish_experiment_lock.acquire()

    #########################
    #    Node Management    #
    #########################

    def connect_to(self, h, p, full=False, force=False):
        """
        Connects a node to another. If learning is running, connections are not allowed (it should be forced).
        Careful, if connection is forced with a new node, it will produce timeouts in the network.

        Args:
            h (str): The host of the node.
            p (int): The port of the node.
            full (bool): If True, the node will be connected to the entire network.
            force (bool): If True, the node will be connected even though it should not be.

        Returns:
            node: The node that has been connected to.
        """
        # Check if learning is running
        if self.round is not None and not force:
            logging.info(
                "[NODE] Cant connect to other nodes when learning is running."
            )
            return None

        # Connect
        return super().connect_to(h, p, full, force)

    def stop(self):
        """
        Stop the node and the learning if it is running.
        """
        if self.round is not None:
            self.__stop_learning()
        self.learner.close()
        super().stop()

    ##########################
    #    Learning Setters    #
    ##########################

    def set_data(self, data):
        """
        Set the data to be used in the learning process (learner).

        Args:
            data: Dataset to be used in the learning process.
        """
        self.learner.set_data(data)

    def set_model(self, model):
        """
        Set the model to use.
        Carefully, model, not only weights.

        Args:
            model: Model to be learned.
        """
        self.learner.set_model(model)

    ###############################################
    #         Network Learning Management         #
    ###############################################

    def set_start_learning(self, rounds=1, epochs=1):
        """
        Start the learning process in the entire network.
        If the node is not the "starting" node, it will wait for the starting node to start the learning process.

        NOTE:
        In first instance, this functionality only is used in one node of the network (defined by the controller).
        After that, the other participants can start this functionality when they receive Events.START_LEARNING_EVENT

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        if self._terminate_flag.is_set():
            logging.info(
                "[NODE] Node must be running to start learning"
            )
            return
        if self.round is None:
            # Start Learning
            logging.info("[NODE] I am the initializer node... | Broadcasting START_LEARNING | Rounds: {} | Epochs: {}".format(rounds, epochs))
            self.broadcast(
                CommunicationProtocol.build_start_learning_msg(rounds, epochs)
            )
            # Initialize model
            self.broadcast(CommunicationProtocol.build_model_initialized_msg())
            self.__wait_init_model_lock.release()
            self.__model_initialized = (
                True
            )
            # Learning Thread
            self.__start_learning_thread(rounds, epochs)
        else:
            logging.info("[NODE] Learning already started")

    def set_stop_learning(self):
        """
        Stop the learning process in the entire network.
        """
        if self.round is not None:
            self.broadcast(CommunicationProtocol.build_stop_learning_msg())
            self.__stop_learning()
        else:
            logging.info("[NODE] Learning already stopped")

    ##################################
    #         Local Learning         #
    ##################################

    def __start_learning_thread(self, rounds, epochs):
        learning_thread = threading.Thread(
            target=self.__start_learning, args=(rounds, epochs)
        )
        learning_thread.name = "learning_thread-" + self.get_name()
        learning_thread.daemon = True
        learning_thread.start()

    def __start_learning(self, rounds, epochs):
        """
        Start the learning process in the local node.

        Args:
            rounds: Number of rounds of the learning process.
            epochs: Number of epochs of the learning process.
        """
        self.__start_thread_lock.acquire()  # Used to avoid create duplicated training threads
        if self.round is None:
            self.round = 0
            self.totalrounds = rounds
            self.learner.init()
            self.__start_thread_lock.release()

            begin = time.time()

            # Send the model parameters (initial model) to neighbors
            self.__gossip_model_difusion(initialization=True)

            # Wait to guarantee new connection heartbeats convergence and fix neighbors
            wait_time = self.config.participant["WAIT_HEARTBEATS_CONVERGENCE"] - (time.time() - begin)
            if wait_time > 0:
                time.sleep(wait_time)
            self.__initial_neighbors = (
                self.get_neighbors()
            )  # used to restore the original list of neighbors after the learning round

            logging.info("[NODE.__start_learning] Learning started in node {} -> Round: {} | Epochs: {}".format(self.get_name(), self.round, epochs))
            self.learner.set_epochs(epochs)
            self.learner.create_trainer()
            self.__train_step()
            logging.info("[NODE.__start_learning] Thread __start_learning finished in node {}".format(self.get_name()))

    def __stop_learning(self):
        """
        Stop the learning process in the local node. Interrupts learning process if it's running.
        """
        logging.info("[NODE] Stopping learning")
        # Rounds
        self.round = None
        self.totalrounds = None
        # Leraner
        self.learner.interrupt_fit()
        # Aggregator
        self.aggregator.check_and_run_aggregation(force=True)
        self.aggregator.set_nodes_to_aggregate([])
        self.aggregator.clear()
        # Try to free wait locks
        try:
            self.__wait_votes_ready_lock.release()
        except threading.ThreadError:
            pass

    ####################################
    #         Model Aggregation         #
    ####################################

    def add_model(self, m):
        """
        Add a model. If the model isn't inicializated, the recieved model is used for it. Otherwise, the model is aggregated using the **aggregator**.

        Args:
            m: Encoded model. Contains model and their contributors
        """
        # Check if Learning is running
        if self.round is not None:
            try:
                if self.__model_initialized:
                    # Add model to aggregator
                    (
                        decoded_model,
                        contributors,
                        weight,
                    ) = self.learner.decode_parameters(m)
                    logging.info("[NODE.add_model] Model received from {} --using--> {} in the other node | Now I add the model using self.aggregator.add_model()".format(contributors, '__gossip_model_diffusion' if contributors is None and weight is None else '__gossip_model_aggregation'))
                    if self.learner.check_parameters(decoded_model):
                        models_added = self.aggregator.add_model(
                            decoded_model, contributors, weight
                        )
                        if models_added is not None:
                            logging.info("[NODE.add_model] self.broadcast with MODELS_AGGREGATED = {}".format(models_added))
                            self.broadcast(
                                CommunicationProtocol.build_models_aggregated_msg(
                                    models_added
                                )
                            )
                    else:
                        raise ModelNotMatchingError("Not matching models")
                else:
                    self.__model_initialized = True
                    logging.info("[NODE] Initialization Model Weights")
                    # Initialize model
                    model, _, _ = self.learner.decode_parameters(m)
                    self.learner.set_parameters(model)
                    self.__wait_init_model_lock.release()
                    self.broadcast(CommunicationProtocol.build_model_initialized_msg())

            except DecodingParamsError as e:
                logging.error("[NODE] Error decoding parameters: " + str(e))
                self.stop()

            except ModelNotMatchingError as e:
                logging.error("[NODE] Models not matching: " + str(e))
                self.stop()

            except Exception as e:
                logging.error("[NODE] Error adding model: " + str(e))
                self.stop()
                raise e
        else:
            logging.error(
                "[NODE] Tried to add a model while learning is not running"
            )

    #######################
    #    Training Steps    #
    #######################

    def __train_step(self):
        """
        Train the model in the local node.
        If the node is in the __train_set list, the training is performed. Otherwise, the node waits for the training to be performed by another node.
        Returns:

        """

        # Set train set
        if self.round is not None:
            for n in self.get_neighbors():
                if n.get_name() not in self.__train_set:
                    self.__train_set.append(n.get_name())
            self.__train_set.append(self.get_name()) if self.get_name() not in self.__train_set else None
            logging.info("[NODE.__train_step] __train_set = {}".format(self.__train_set))
            self.__validate_train_set()

        # TODO: Improve in the future
        # is_train_set = self.get_name() in self.__train_set
        is_train_set = True
        #if is_train_set and (self.config.participant["device_args"]["role"] == Role.AGGREGATOR or self.config.participant["device_args"]["role"] == Role.SERVER):
        if is_train_set and (self.config.participant["device_args"]["role"] == Role.AGGREGATOR):
            logging.info("[NODE.__train_step] Role.AGGREGATOR/Role.SERVER process...")
            # Full connect train set
            if self.round is not None:
                self.__connect_and_set_aggregator()

            # Evaluate and send metrics
            if self.round is not None:
                self.__evaluate()

            # Train
            # if self.round is not None and self.config.participant["device_args"]["role"] != Role.SERVER:
            if self.round is not None:
                self.__train()

            # Aggregate Model
            if self.round is not None:
                logging.info("[NODE.__train_step] self.aggregator.add_model with MY MODEL")
                self.aggregator.add_model(
                    self.learner.get_parameters(),
                    [self.get_name()],
                    self.learner.get_num_samples()[0],
                )
                logging.info("[NODE.__train_step] self.broadcast with MODELS_AGGREGATED = MY_NAME")
                self.broadcast(
                    CommunicationProtocol.build_models_aggregated_msg([self.get_name()])
                )
                # if self.config.participant["device_args"]["role"] == Role.SERVER:
                #     self.__gossip_model_difusion()
                # else:
                self.__gossip_model_aggregation()

        # elif self.config.participant["device_args"]["role"] == Role.TRAINER:
        #     logging.info("[NODE.__train_step] Role.TRAINER process...")
        #     logging.info("[NODE.__train_step] __waiting_aggregated_model = {}".format(self.aggregator.get_waiting_aggregated_model()))
        #     if self.round is not None:
        #         self.__connect_and_set_aggregator()

        #     # Evaluate and send metrics
        #     if self.round is not None:
        #         self.__evaluate()

        #     # Train
        #     if self.round is not None:
        #         self.__train()

        #     # Aggregate Model
        #     if self.round is not None:
        #         logging.info("[NODE.__train_step] self.aggregator.add_model with MY MODEL")
        #         # Node has to aggregate its own model before sending it to the aggregator
        #         self.aggregator.add_model(
        #             self.learner.get_parameters(),
        #             [self.get_name()],
        #             self.learner.get_num_samples()[0],
        #         )

        #         logging.info("[NODE.__train_step] self.broadcast with MODELS_AGGREGATED = MY_NAME")
        #         self.broadcast(
        #             CommunicationProtocol.build_models_aggregated_msg([self.get_name()])
        #         )

        #         self.__gossip_model_aggregation()

        #         self.aggregator.set_waiting_aggregated_model()
        #         time.sleep(5)

        elif self.config.participant["device_args"]["role"] == Role.PROXY:
            # If the node is a proxy, it stores the parameters received from the neighbors.
            # When the node reaches a timeout or the number of parameters received is equal to a specific number, it will send the parameters to all the neighbors.
            logging.info("[NODE.__train_step] Role.PROXY process...")

            # Aggregate Model
            if self.round is not None:
                # self.aggregator.add_model(
                #    self.learner.get_parameters(),
                #    [self.get_name()],
                #    self.learner.get_num_samples()[0],
                # )

                self.broadcast(
                    CommunicationProtocol.build_models_aggregated_msg([self.get_name()])
                )
                # Timeout to send the parameters to the neighbors?
                if datetime.now() > self.__timeout:
                    logging.info("[NODE.__train_step (PROXY)] Timeout reached. Sending parameters to neighbors...")
                    self.__gossip_model_aggregation()
                    self.__timeout = datetime.now() + timedelta(seconds=10)
                # The proxy node sets the waiting aggregated model flag to True
                # In this case, the proxy waits for params and add them to the local storage
                self.aggregator.set_waiting_aggregated_model()

        elif self.config.participant["device_args"]["role"] == Role.IDLE:
            # Role.IDLE functionality

            # Set Models To Aggregate
            # Node won't participate in aggregation process.
            # __waiting_aggregated_model = True
            # Then, when the node receives a PARAMS_RECEIVED_EVENT, it will run add_model, and it set parameters to the model
            self.aggregator.set_waiting_aggregated_model()

        else:
            logging.warning("[NODE.__train_step] Role not implemented yet")

        # Gossip aggregated model
        # if self.round is not None:
        #    self.__gossip_model_difusion()

        # Finish round
        if self.round is not None:
            self.__on_round_finished()

    def __validate_train_set(self):
        # Verify if node set is valid (can happend that a node was down when the votes were being processed)
        for tsn in self.__train_set:
            if tsn not in self.get_network_nodes():
                if tsn != self.get_name():
                    self.__train_set.remove(tsn)

        logging.info(
            "[NODE.__validate_train_set] Train set of {} nodes: {}".format(
                len(self.__train_set), self.__train_set
            )
        )

    ##########################
    #    Connect Trainset    #
    ##########################

    def __connect_and_set_aggregator(self):
        # Set Models To Aggregate
        self.aggregator.set_nodes_to_aggregate(self.__train_set)
        logging.info("[NODE.__connect_and_set_aggregator] Aggregator set to: {}".format(self.__train_set))

    ############################
    #    Train and Evaluate    #
    ############################

    def __train(self):
        logging.info("[NODE.__train] Start training...")
        print("[NODE.__train] Start training...")
        self.learner.fit()
        logging.info("[NODE.__train] Finish training...")
        print("[NODE.__train] Finish training...")

    def __evaluate(self):
        logging.info("[NODE.__evaluate] Start evaluation...")
        print("[NODE.__evaluate] Start evaluation...")
        self.learner.evaluate()
        logging.info("[NODE.__evaluate] Finish evaluation...")
        print("[NODE.__evaluate] Finish evaluation...")

    ######################
    #    Round finish    #
    ######################

    def __on_round_finished(self):
        # Remove trainset connections
        for nc in self.get_neighbors():
            if nc not in self.__initial_neighbors:
                self.rm_neighbor(nc)
        # Set Next Round
        self.aggregator.clear()
        logging.info("[NODE] Finalizing round: {}".format(self.round))
        self.learner.finalize_round()  # TODO: Fix to improve functionality
        
        logging.info(f"[NODE] Saving model in round {self.round}")
        self.learner.save_model(round=self.round)
        
        self.round = self.round + 1
        ###FER###
        #self.learner.logger.log_metrics({"Round": self.round}, step=self.learner.logger.global_step)
        ###FER###
        logging.info("[LightningLearner] Starting round: {}".format(self.round))
        # Clear node aggregation
        for nc in self.get_neighbors():
            nc.clear_models_aggregated()      

        # Next Step or Finish
        logging.info(
            "[NODE] Round {} of {} finished.".format(
                self.round, self.totalrounds
            )
        )
        if self.round < self.totalrounds:
            self.__train_step()
        else:
            logging.debug("[NODE] FL finished | Models aggregated = {}".format([nc.get_models_aggregated() for nc in self.get_neighbors()]))
            # At end, all nodes compute metrics
            self.__evaluate()
            # Finish
            logging.info(
                "[NODE] FL experiment finished | Round: {} | Total rounds: {} | [!] Both to None".format(
                    self.round, self.totalrounds
                )
            )
            self.round = None
            self.totalrounds = None
            self.__model_initialized = False
            logging.info("[NODE] FL experiment finished | __stop_learning()")
            self.__stop_learning()

    # def __transfer_aggregator_role(self, schema):
    #     if schema == "random":
    #         logging.info("[NODE.__transfer_aggregator_role] Transferring aggregator role using schema {}".format(schema))
    #         # Random
    #         nc = random.choice(self.get_neighbors())
    #         msg = CommunicationProtocol.build_transfer_leadership_msg()
    #         nc.send(msg)
    #         self.config.participant['device_args']["role"] = "trainer"
    #         logging.info("[NODE.__transfer_aggregator_role] Aggregator role transfered to {}.".format(nc.get_name()))
    #     else:
    #         logging.info("[NODE.__transfer_aggregator_role] Schema {} not found.".format(schema))

    #########################
    #    Model Gossiping    #
    #########################

    def __gossip_model_aggregation(self):
        logging.info("[NODE.__gossip_model_aggregation] Gossiping...")
        # Anonymous functions
        candidate_condition = lambda nc: nc.get_name() in self.__train_set and len(nc.get_models_aggregated()) < len(self.__train_set)
        status_function = lambda nc: (nc.get_name(), len(nc.get_models_aggregated()))
        model_function = lambda nc: self.aggregator.get_partial_aggregation(nc.get_models_aggregated())

        # Gossip
        self.__gossip_model(candidate_condition, status_function, model_function)

    def __gossip_model_difusion(self, initialization=False):
        logging.info("[NODE.__gossip_model_difusion] Gossiping...")
        # Send model parameters using gossiping
        # Wait a model (init or aggregated)
        if initialization:
            self.__wait_init_model_lock.acquire()
            logging.info("[NODE.__gossip_model_difusion] Initialization=True")
            candidate_condition = lambda nc: not nc.get_model_initialized()
        else:
            self.__finish_aggregation_lock.acquire()
            logging.info("[NODE.__gossip_model_difusion] Initialization=False")
            candidate_condition = lambda nc: nc.get_model_ready_status() < self.round

        # Anonymous functions
        status_function = lambda nc: nc.get_name()
        model_function = lambda _: (
            self.learner.get_parameters(),
            None,
            None,
        )  # At diffusion, contributors are not relevant

        # Gossip
        self.__gossip_model(candidate_condition, status_function, model_function)

    def __gossip_model(self, candidate_condition, status_function, model_function):
        logging.debug("[NODE.__gossip_model] Traceback", stack_info=True)
        # Initialize list with status of nodes in the last X iterations
        last_x_status = []
        j = 0

        while True:
            # Get time to calculate frequency
            begin = time.time()

            # If the trainning has been interrupted, stop waiting
            if self.round is None:
                logging.info(
                    "[NODE] Stopping model gossip process.")
                return

            # Get nodes which need models
            logging.info("---------------------Feedback about neighbors---------------------")
            logging.info("[NODE.__gossip_model] Neighbors: {}".format(self.get_neighbors()))
            for nc in self.get_neighbors():
                logging.info("---------------------Feedback about neighbor {}---------------------".format(nc.name))
                logging.info("[NODE.__gossip_model] Neighbor: {} | My __train_set: {} | Nc.modelsaggregated: {}".format(nc.name, self.__train_set, nc.get_models_aggregated()))
                logging.info("[NODE.__gossip_model] Neighbor: {} | Candidate_condition return: {}".format(nc.name, candidate_condition(nc)))
                logging.info("[NODE.__gossip_model] Neighbor: {} | Status_function return: {}".format(nc.name, status_function(nc)))
                logging.info("---------------------End of feedback about neighbor {}---------------------".format(nc.name))
            logging.info("------------------------------------------------------------------")

            nei = [nc for nc in self.get_neighbors() if candidate_condition(nc)]
            logging.info("[NODE.__gossip_model] Selected (to exclude) based on condition: {}".format(nei))

            # Determine end of gossip
            if not nei:
                logging.info("[NODE] Gossip finished.")
                return

            # Save state of neighbors. If nodes are not responding gossip will stop
            if len(last_x_status) != self.config.participant["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]:
                last_x_status.append([status_function(nc) for nc in nei])
            else:
                last_x_status[j] = str([status_function(nc) for nc in nei])
                j = (j + 1) % self.config.participant["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]

                # Check if las messages are the same
                for i in range(len(last_x_status) - 1):
                    if last_x_status[i] != last_x_status[i + 1]:
                        break
                    logging.info(
                        "[NODE] Gossiping exited for {} equal rounds.".format(
                            self.config.participant["GOSSIP_EXIT_ON_X_EQUAL_ROUNDS"]
                        )
                    )
                    return

            # Select a random subset of neighbors
            samples = min(self.config.participant["GOSSIP_MODELS_PER_ROUND"], len(nei))
            nei = random.sample(nei, samples)
            logging.info("[NODE.__gossip_model] Selected a random subset of neighbors (to exclude): {}".format(nei))

            # Generate and Send Model Partial Aggregations (model, node_contributors)
            for nc in nei:
                model, contributors, weights = model_function(nc)
                # Send Partial Aggregation
                if model is not None:
                    logging.info(
                        "[NODE] Gossiping model to {}.".format(
                            nc.get_name()
                        )
                    )
                    encoded_model = self.learner.encode_parameters(
                        params=model, contributors=contributors, weight=weights
                    )
                    logging.info("[NODE.__gossip_model] Building params message | Contributors: {}".format(contributors))
                    encoded_msgs = CommunicationProtocol.build_params_msg(encoded_model, self.config.participant["BLOCK_SIZE"])
                    logging.info("[NODE.__gossip_model] Sending params message to {}".format(nc))
                    # Send Fragments
                    for msg in encoded_msgs:
                        nc.send(msg)
                else:
                    logging.info("[NODE.__gossip_model] Model returned by model_function is None")
            # Wait to guarantee the frequency of gossipping
            time_diff = time.time() - begin
            time_sleep = 1 / self.config.participant["GOSSIP_MODELS_FREC"] - time_diff
            if time_sleep > 0:
                time.sleep(time_sleep)

    ###########################
    #     Observer Events     #
    ###########################

    def update(self, event, obj):
        """
        Observer update method. Used to handle events that can occur in the different components and connections of the node.

        Args:
            event (Events): Event that has occurred.
            obj: Object that has been updated.
        """
        if len(str(obj)) > 300:
            logging.debug("[NODE.update (observer)] Event that has occurred: {} | Obj information: Too long [...]".format(event))
        else:
            logging.debug("[NODE.update (observer)] Event that has occurred: {} | Obj information: {}".format(event, obj))

        if event == Events.NODE_CONNECTED_EVENT:
            n, force = obj
            if self.round is not None and not force:
                logging.info(
                    "[NODE] Cant connect to other nodes when learning is running. (however, other nodes can be connected to the node.)"
                )
                n.stop()
                return

        elif event == Events.SEND_ROLE_EVENT:
            self.broadcast(CommunicationProtocol.build_role_msg(self.get_name(), self.config.participant["device_args"]["role"]))

        elif event == Events.ROLE_RECEIVED_EVENT:
            # Update the heartbeater with the role node
            # obj = (node_name, role)
            self.heartbeater.add_node_role(obj[0], obj[1])

        elif event == Events.AGGREGATION_FINISHED_EVENT:
            # Set parameters and communate it to the training process
            if obj is not None:
                logging.info("[NODE.update] Override the local model with obj received")
                self.learner.set_parameters(obj)
                # Share that aggregation is done
                self.broadcast(CommunicationProtocol.build_models_ready_msg(self.round))
            else:
                logging.error(
                    "[NODE] Aggregation finished with no parameters"
                )
            try:
                self.__finish_aggregation_lock.release()
                logging.info("[NODE.__finish_aggregation_lock] __finish_aggregation_lock.release()")
            except threading.ThreadError:
                pass

        elif event == Events.START_LEARNING_EVENT:
            self.__start_learning_thread(obj[0], obj[1])

        elif event == Events.STOP_LEARNING_EVENT:
            self.__stop_learning()

        elif event == Events.PARAMS_RECEIVED_EVENT:
            logging.info("[NODE] Params received")
            self.add_model(obj)

        elif event == Events.METRICS_RECEIVED_EVENT:
            # name, round, loss, metric = obj
            # self.learner.log_validation_metrics(loss, metric, round=round, name=name)
            pass

        elif event == Events.TRAIN_SET_VOTE_RECEIVED_EVENT:
            node, votes = obj
            self.__train_set_votes_lock.acquire()
            self.__train_set_votes[node] = votes
            self.__train_set_votes_lock.release()
            # Communicate to the training process that a vote has been received
            try:
                self.__wait_votes_ready_lock.release()
            except threading.ThreadError:
                pass

        elif event == Events.STORE_MODEL_PARAMETERS_EVENT:
            if obj is not None:
                logging.info("[NODE.update] Store the model parameters received")
                self.__store_model_parameters(obj)
                # Share that aggregation is done
                self.broadcast(CommunicationProtocol.build_models_ready_msg(self.round))
            else:
                logging.error(
                    "[NODE] Error storing the model parameters"
                )
                self.stop()
            try:
                self.__finish_aggregation_lock.release()
                logging.info("[NODE.__finish_aggregation_lock] __finish_aggregation_lock.release()")
            except threading.ThreadError:
                pass

        # Execute BaseNode update
        super().update(event, obj)

    def __store_model_parameters(self, obj):
        """
        Store the model parameters in the node.

        Args:
            obj: Model parameters.

        Returns:

        """
        self.__stored_model_parameters += obj
        logging.info("[NODE.__store_model_parameters (PROXY)] Stored model parameters: {}".format(len(self.__stored_model_parameters)))
