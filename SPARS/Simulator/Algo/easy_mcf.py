from .easy_normal import EASYNormal
import networkx as nx
import logging

logger = logging.getLogger("mcf_power")

class EASYMinimumCostFlow(EASYNormal):
    def __init__(self, state, waiting_queue, start_time, jobs_manager, 
                 timeout=None, platform_info=None, workload_info=None,
                 recompute_interval=3600, lookahead_window=7200):
        """
        EASY Backfilling with Minimum-Cost Flow based power management.
        Based on "On Energy Conservation in Data Centers" by Albers (2019).
        
        Args:
            recompute_interval: How often to recompute the schedule (seconds)
            lookahead_window: How far ahead to look when computing schedule (seconds)
        """
        super().__init__(state, waiting_queue, start_time, jobs_manager, timeout=None)
        
        self.platform_info = platform_info
        self.workload_info = workload_info
        self.recompute_interval = recompute_interval
        self.lookahead_window = lookahead_window
        
        self.last_computation_time = -float('inf')
        self.power_schedule = {}
        self.last_event_time = -1
        
        self.node_id_to_index = {}
        self.index_to_node_id = {}
        if platform_info:
            for i, machine in enumerate(platform_info['machines']):
                self.node_id_to_index[machine['id']] = i
                self.index_to_node_id[i] = machine['id']
        
        self._recompute_schedule()

    def _recompute_schedule(self):
        """Recompute the optimal power schedule based on current state."""
        if not self.platform_info or not self.workload_info:
            logger.warning("Missing platform or workload info, skipping power schedule computation")
            return
            
        logger.info(f"Recomputing power schedule at time {self.current_time}")
        
        # generate demand profile considering current state
        demand_profile, time_points = self._generate_demand_profile_adaptive()
        
        if not time_points or len(time_points) < 2:
            logger.warning("Not enough time points in workload")
            self.power_schedule = {}
            return
        
        # build the min-cost flow network
        G = self._build_network(self.platform_info, demand_profile, time_points)
        
        # well compute mcf
        try:
            flow_dict = nx.min_cost_flow(G)
            flow_cost = nx.cost_of_flow(G, flow_dict)
            logger.info(f"Initial min-cost flow computed with cost: {flow_cost}")
            
            # make the flow consistent (critical step from paper, will kill ur pc perhaps)
            flow_dict = self._make_flow_consistent(flow_dict, len(self.platform_info['machines']), 
                                                   len(time_points), time_points)
            
            self.power_schedule = self._extract_schedule_from_flow(flow_dict, 
                                                                  len(self.platform_info['machines']), 
                                                                  len(time_points), time_points)
            
            self.last_computation_time = self.current_time
            logger.info(f"Power schedule updated with {len(self.power_schedule)} time points")
            
        except (nx.NetworkXUnfeasible, nx.NetworkXError) as e:
            logger.error(f"Failed to compute min-cost flow: {e}")
            self.power_schedule = {}

    def _generate_demand_profile_adaptive(self):
        """
        Generate demand profile considering current jobs and system state.
        This version is aware of the current simulation state.
        """
        # get future jobs within lookahead window
        # end_time = self.current_time + self.lookahead_window # turn this on if u want to turn on batching
        future_jobs = [j for j in self.workload_info['jobs'] 
                      if j['subtime'] >= self.current_time]
        
        # include currently waiting and scheduled jobs
        current_demand = len(self.scheduled) + len([j for j in self.waiting_queue 
                                                    if j['job_id'] not in self.scheduled])
        
        # combine all jobs for demand calculation
        all_relevant_jobs = list(self.waiting_queue) + future_jobs
        
        return self._calculate_demand_from_jobs(all_relevant_jobs, current_demand)
    
    def _calculate_demand_from_jobs(self, jobs, initial_demand):
        """
        Calculate demand profile from a set of jobs using EASY backfilling logic.
        """
        jobs = sorted(jobs, key=lambda j: j.get('subtime', 0))
        num_nodes = len(self.platform_info['machines'])
        
        # track node availability (considering current allocations)
        node_release_times = [0] * num_nodes
        for i, node in enumerate(self.state):
            if node['job_id'] is not None:
                # node is currently busy, estimate release time
                node_release_times[i] = self.current_time + 1800  # default estimate
        
        events = []  # [(time, resource_change)]
        events.append((self.current_time, initial_demand))  # start with curr demand
        
        for job in jobs:
            res = job.get('res', 0)
            if res <= 0 or res > num_nodes:
                continue
            
            # find earliest start time for this job (easy backfilling logic)
            available_nodes = sorted(range(num_nodes), key=lambda i: node_release_times[i])
            required_nodes = available_nodes[:res]
            
            # so if all needed nodes are free the job will start
            start_time = max(job.get('subtime', self.current_time), 
                           max(node_release_times[i] for i in required_nodes))
            
            # use requested time as duration estimate
            duration = job.get('reqtime', job.get('runtime', 1800))
            finish_time = start_time + duration
            
            for i in required_nodes:
                node_release_times[i] = finish_time
            
            events.append((start_time, res))
            events.append((finish_time, -res))
        
        events.sort()
        
        # create demand profile
        time_points = sorted(list(set([e[0] for e in events if e[0] >= self.current_time])))
        if self.current_time not in time_points:
            time_points.insert(0, self.current_time)
        
        demand_profile = {}
        current_demand = 0
        event_idx = 0
        
        for i in range(len(time_points)):
            t_k = time_points[i]
            
            # process all events at this time
            while event_idx < len(events) and events[event_idx][0] <= t_k:
                if events[event_idx][0] == t_k:
                    current_demand += events[event_idx][1]
                event_idx += 1
            
            # so demand wont exceed available nodes
            current_demand = max(0, min(current_demand, num_nodes))
            demand_profile[t_k] = current_demand
        
        return demand_profile, time_points

    def _build_network(self, platform_info, demand_profile, time_points):
        """Builds the min-cost flow graph based on Albers (2019), Figure 1 & 2[cite: 191, 236]."""
        G = nx.DiGraph()
        m = len(platform_info['machines'])
        n = len(time_points)
        
        # add main source (a0) and sink (b0) [cite: 247]
        G.add_node("a0", demand=-m)
        G.add_node("b0", demand=m)

        # add demand sources (ak) and sinks (bk) for each interval [cite: 257]
        for k in range(n - 1):
            d_k = demand_profile.get(time_points[k], 0)
            if d_k > 0:
                G.add_node(f"ak{k}", demand=-d_k)
                G.add_node(f"bk{k}", demand=d_k)
        
        for i, machine in enumerate(platform_info['machines']):
            active_power = machine['dvfs_profiles']['base']['power']
            sleep_power = machine['states']['sleeping']['power']
            
            switch_on_time = machine['states']['switching_on']['transitions'][0]['transition_time']
            switch_on_power = machine['states']['switching_on']['power']
            switch_off_time = machine['states']['switching_off']['transitions'][0]['transition_time']
            switch_off_power = machine['states']['switching_off']['power']
            
            # power-up energy (delta_i in paper)
            delta_i = switch_on_time * switch_on_power
            
            # add nodes for this server's component
            for k in range(n):
                G.add_node(f"u_{i},{k}") # upper path node (active state)
                G.add_node(f"l_{i},{k}") # lower path node (sleep state)
                if k < n - 1:
                    G.add_node(f"la_{i},{k}") # lower path auxiliary 'a'
                    G.add_node(f"lb_{i},{k}") # lower path auxiliary 'b'
            
            # connect to main source/sink
            G.add_edge("a0", f"u_{i},0", capacity=1, weight=0)
            G.add_edge("a0", f"l_{i},0", capacity=1, weight=0)
            G.add_edge(f"u_{i},{n-1}", "b0", capacity=1, weight=0)
            G.add_edge(f"l_{i},{n-1}", "b0", capacity=1, weight=0)
            
            # add edges for each time interval
            for k in range(n - 1):
                interval = time_points[k+1] - time_points[k]
                if interval <= 0:
                    continue
                
                # upper path edge (active state cost)
                active_cost = (active_power - sleep_power) * interval
                G.add_edge(f"u_{i},{k}", f"u_{i},{k+1}", capacity=1, weight=active_cost)
                
                # lower path edges (sleep state - base cost already subtracted)
                G.add_edge(f"l_{i},{k}", f"la_{i},{k}", capacity=1, weight=0)
                G.add_edge(f"la_{i},{k}", f"lb_{i},{k}", capacity=1, weight=0)
                G.add_edge(f"lb_{i},{k}", f"l_{i},{k+1}", capacity=1, weight=0)
                
                # state transition edges
                G.add_edge(f"l_{i},{k}", f"u_{i},{k}", capacity=1, weight=delta_i)  # power-up
                G.add_edge(f"u_{i},{k+1}", f"l_{i},{k+1}", capacity=1, weight=0)    # power-down (free for now, ill set it later after i ask santana)
                
                # demand enforcement edges
                d_k = demand_profile.get(time_points[k], 0)
                if d_k > 0:
                    G.add_edge(f"ak{k}", f"la_{i},{k}", capacity=1, weight=0)
                    G.add_edge(f"lb_{i},{k}", f"bk{k}", capacity=1, weight=0)
        
        return G

    def _make_flow_consistent(self, flow_dict, m, n, time_points):
        """
        Modify flow to ensure it corresponds to a valid schedule.
        This implements the flow modification procedure from Section 3.2 of Albers (2019).
        
        Key insight: An arbitrary min-cost flow may have a server in both upper
        and lower paths simultaneously, which is physically impossible.
        """
        modified_flow = {}
        
        for i in range(m):
            # track which path the server takes through time
            server_path = []  # list of 'upper' or 'lower' for each time point
            
            # determine initial path from source
            if flow_dict.get("a0", {}).get(f"u_{i},0", 0) > 0.5:
                current_path = 'upper'
            else:
                current_path = 'lower'
            server_path.append(current_path)
            
            # trace path through the network
            for k in range(n - 1):
                # check for state transitions
                if current_path == 'lower':
                    # check for power-up transition
                    if flow_dict.get(f"l_{i},{k}", {}).get(f"u_{i},{k}", 0) > 0.5:
                        current_path = 'upper'
                else:
                    # check for power-down transition
                    if flow_dict.get(f"u_{i},{k+1}", {}).get(f"l_{i},{k+1}", 0) > 0.5:
                        current_path = 'lower'
                
                server_path.append(current_path)
            
            # rebuild consistent flow for this server
            for k in range(n):
                if server_path[k] == 'upper':
                    # flow goes through upper path
                    if k == 0:
                        modified_flow.setdefault("a0", {})[f"u_{i},0"] = 1
                    if k < n - 1:
                        modified_flow.setdefault(f"u_{i},{k}", {})[f"u_{i},{k+1}"] = 1
                    if k == n - 1:
                        modified_flow.setdefault(f"u_{i},{n-1}", {})["b0"] = 1
                    
                    # add transitions
                    if k > 0 and server_path[k-1] == 'lower':
                        # power-up transition
                        modified_flow.setdefault(f"l_{i},{k}", {})[f"u_{i},{k}"] = 1
                    if k < n - 1 and server_path[k+1] == 'lower':
                        # power-down transition
                        modified_flow.setdefault(f"u_{i},{k+1}", {})[f"l_{i},{k+1}"] = 1
                
                else:
                    # lower path
                    if k == 0:
                        modified_flow.setdefault("a0", {})[f"l_{i},0"] = 1
                    if k < n - 1:
                        modified_flow.setdefault(f"l_{i},{k}", {})[f"la_{i},{k}"] = 1
                        modified_flow.setdefault(f"la_{i},{k}", {})[f"lb_{i},{k}"] = 1
                        modified_flow.setdefault(f"lb_{i},{k}", {})[f"l_{i},{k+1}"] = 1
                        
                        # handle demand edges if server is in lower path
                        d_k = flow_dict.get(f"ak{k}", {}).get(f"la_{i},{k}", 0)
                        if d_k > 0:
                            modified_flow.setdefault(f"ak{k}", {})[f"la_{i},{k}"] = 1
                            modified_flow.setdefault(f"lb_{i},{k}", {})[f"bk{k}"] = 1
                    
                    if k == n - 1:
                        modified_flow.setdefault(f"l_{i},{n-1}", {})["b0"] = 1
        
        return modified_flow

    def _extract_schedule_from_flow(self, flow_dict, m, n, time_points):
        """
        Extract power state transition schedule from the consistent flow.
        Returns a dictionary mapping timestamps to power events.
        """
        power_schedule = {}
        
        for k in range(n):
            if k >= len(time_points):
                continue
                
            t_k = time_points[k]
            nodes_to_power_up = []
            nodes_to_power_down = []
            
            # check each server for state transitions at this time
            for i in range(m):
                node_id = self.index_to_node_id.get(i, i)  # map to actual node id
                
                # check for powerup
                if k > 0 and flow_dict.get(f"l_{i},{k}", {}).get(f"u_{i},{k}", 0) > 0.5:
                    nodes_to_power_up.append(node_id)
                
                # check for powerdown
                if k < n - 1 and flow_dict.get(f"u_{i},{k+1}", {}).get(f"l_{i},{k+1}", 0) > 0.5:
                    # power-down happens at the end of interval k (start of k+1)
                    nodes_to_power_down.append(node_id)
            
            # add events to schedule
            if nodes_to_power_up:
                power_schedule.setdefault(t_k, []).append({
                    'type': 'switch_on',
                    'nodes': sorted(list(set(nodes_to_power_up)))
                })
            
            if nodes_to_power_down:
                # power-down events happen at the end of the interval
                next_time = time_points[k+1] if k+1 < len(time_points) else t_k
                power_schedule.setdefault(next_time, []).append({
                    'type': 'switch_off', 
                    'nodes': sorted(list(set(nodes_to_power_down)))
                })
        
        return power_schedule

    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        """
        Main scheduling function that integrates EASY backfilling with 
        optimal power management.
        """
        # update state for potential recomputation
        self.state = new_state
        self.waiting_queue = waiting_queue
        
        # standard easy backfilling scheduling
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)
        super().FCFSNormal() # well if u wonder why dont inherent easy directly bcs we dont need the timeout_policy
        self.backfill()
        
        # recompute power schedule if needed (rolling horizon)
        #if self.current_time - self.last_computation_time > self.recompute_interval: # turn this on if u want to turn on batching

        #    self._recompute_schedule()
        
        # apply pre-computed power management decisions
        if self.current_time > self.last_event_time:
            self._apply_power_events()
            self.last_event_time = self.current_time
        
        return self.events

    def _apply_power_events(self):
        """
        Apply power state transitions from the pre-computed schedule.
        Ensures safety by checking current node states.
        """
        if self.current_time not in self.power_schedule:
            return
        
        # build lookup tables for current state
        state_by_id = {node['id']: node for node in self.state}
        allocated_ids = {node['id'] for node in self.allocated}
        scheduled_ids = set(self.scheduled)
        
        for event in self.power_schedule[self.current_time]:
            nodes_to_action = []
            
            if event['type'] == 'switch_on':
                # only switch on nodes that are actually sleeping
                for node_id in event['nodes']:
                    node = state_by_id.get(node_id)
                    if node and node['state'] == 'sleeping':
                        nodes_to_action.append(node_id)
                
                if nodes_to_action:
                    logger.debug(f"Switching on nodes at {self.current_time}: {nodes_to_action}")
                    super().push_event(self.current_time, {
                        'type': 'switch_on',
                        'nodes': nodes_to_action
                    })
            
            elif event['type'] == 'switch_off':
                # only switch off idle, active nodes that arent allocated
                for node_id in event['nodes']:
                    node = state_by_id.get(node_id)
                    if (node and 
                        node['state'] == 'active' and 
                        node['job_id'] is None and 
                        node_id not in allocated_ids and
                        not node.get('reserved', False)):
                        nodes_to_action.append(node_id)
                
                if nodes_to_action:
                    logger.debug(f"Switching off nodes at {self.current_time}: {nodes_to_action}")
                    super().push_event(self.current_time, {
                        'type': 'switch_off',
                        'nodes': nodes_to_action
                    })
