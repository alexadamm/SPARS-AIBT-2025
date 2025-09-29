from .easy_normal import EASYNormal
import networkx as nx

class EASYMinimumCostFlow(EASYNormal):
    def __init__(self, state, waiting_queue, start_time, jobs_manager, timeout=None, platform_info=None, workload_info=None):
        super().__init__(state, waiting_queue, start_time, jobs_manager, timeout)
        self.power_schedule = self._compute_optimal_schedule(platform_info, workload_info)
        self.last_event_time = -1

    def _compute_optimal_schedule(self, platform_info, workload_info):
        """
        Calculates the optimal power state schedule using a min-cost flow approach.
        This is an offline, pre-computation step that runs once before the simulation starts.
        The methodology is based on the Albers (2019) paper for servers with two states.
        """
        # generate demand profile from workload_info
        demand_profile, time_points = self._generate_demand_profile(workload_info, platform_info)
        if not time_points or len(time_points) < 2:
            print("Not enough time points in workload. No power schedule to compute.")
            return {}
    
        # buildd it to the min-cost flow network
        G = self._build_network(platform_info, demand_profile, time_points)
    
        # compute mcf
        try:
            # nx.min_cost_flow returns only the flow dictionary.
            flow_dict = nx.min_cost_flow(G)
            # the cost must be calculated separately.
            flow_cost = nx.cost_of_flow(G, flow_dict)
            print(f"Initial min-cost flow computed with cost: {flow_cost}")
        except nx.NetworkXUnfeasible:
            print("Error: Could not solve the min-cost flow problem. The problem may be unfeasible.")
            return {}
        except nx.NetworkXError as e:
            print(f"An error occurred during min-cost flow computation: {e}")
            return {}
    
        # TODO: make the flow consistent (as per the paper)
        # its crucial because an arbitrary mcf doesnt always
        # correspond to a valid schedule. For now, we proceed with the direct output.
        
        # extratc the final power on/off schedule
        optimal_schedule = self._extract_schedule_from_flow(flow_dict, len(platform_info['machines']), len(time_points), time_points)
        
        print("optimal power schedule computed.")
        return optimal_schedule

    def _generate_demand_profile(self, workload_info, platform_info):
        """
        Generates a discrete demand profile (d_k servers needed at time t_k)
        by running a greedy, simplified simulation of job placements.
        """
        jobs = sorted(workload_info['jobs'], key=lambda j: j['subtime'])
        num_nodes = len(platform_info['machines'])
        node_release_times = [0] * num_nodes
        
        events = [] # [(time, resource_change)]

        for job in jobs:
            if job['res'] <= 0 or job['res'] > num_nodes:
                continue # skip invalid jobs

            # find the earliest time the job can start
            # sort nodes by their release time to greedily pick the soonest available
            available_nodes_indices = sorted(range(num_nodes), key=lambda i: node_release_times[i])
            required_nodes_indices = available_nodes_indices[:job['res']]
            
            # maximum of the job's submission time and when the last required node becomes free
            start_time = max(job['subtime'], max(node_release_times[i] for i in required_nodes_indices))
            finish_time = start_time + job['reqtime']

            for i in required_nodes_indices:
                node_release_times[i] = finish_time
                
            events.append((start_time, job['res']))
            events.append((finish_time, -job['res']))
            
        events.sort()
        
        # create demand profile {t_k: d_k} and a sorted list of time points
        demand_profile = {}
        time_points = sorted(list(set([e[0] for e in events] + [0])))
        
        current_demand = 0
        event_idx = 0
        for i in range(len(time_points)):
            t_k = time_points[i]
            # set demand for the interval preceding this time point
            if i > 0:
                demand_profile[time_points[i-1]] = current_demand
            
            # update demand by processing all events at the current time point
            while event_idx < len(events) and events[event_idx][0] == t_k:
                current_demand += events[event_idx][1]
                event_idx += 1
        
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
            r_i = machine['dvfs_profiles']['base']['power']
            
            # calculate power-up energy (delta_i) [cite: 245]
            on_time = machine['states']['switching_on']['transitions'][0]['transition_time']
            on_power = machine['states']['switching_on']['power']
            delta_i = on_time * on_power

            # add component nodes for server i [cite: 191]
            for k in range(n):
                G.add_node(f"u_{i},{k}") # upper path node (active state)
                G.add_node(f"l_{i},{k}") # lower path node (sleep state)
                if k < n - 1:
                    G.add_node(f"la_{i},{k}") # lower path auxiliary 'a'
                    G.add_node(f"lb_{i},{k}") # lower path auxiliary 'b'
            
            G.add_edge("a0", f"l_{i},0", capacity=1, weight=0)
            G.add_edge(f"l_{i},{n-1}", "b0", capacity=1, weight=0)
            
            for k in range(n - 1):
                interval = time_points[k+1] - time_points[k]
                if interval <= 0: continue
                
                # upper path edge (active state) [cite: 239]
                G.add_edge(f"u_{i},{k}", f"u_{i},{k+1}", capacity=1, weight=r_i * interval)
                
                # lower path edges (sleep state) [cite: 242]
                G.add_edge(f"l_{i},{k}", f"la_{i},{k}", capacity=1, weight=0)
                G.add_edge(f"la_{i},{k}", f"lb_{i},{k}", capacity=1, weight=0)
                G.add_edge(f"lb_{i},{k}", f"l_{i},{k+1}", capacity=1, weight=0)
                
                # state transition edges [cite: 244]
                G.add_edge(f"l_{i},{k}", f"u_{i},{k}", capacity=1, weight=delta_i) # power-up
                G.add_edge(f"u_{i},{k+1}", f"l_{i},{k+1}", capacity=1, weight=0)   # power-down
                
                # demand enforcement edges [cite: 257, 258]
                d_k = demand_profile.get(time_points[k], 0)
                if d_k > 0:
                    G.add_edge(f"ak{k}", f"la_{i},{k}", capacity=1, weight=0)
                    G.add_edge(f"lb_{i},{k}", f"bk{k}", capacity=1, weight=0)
        return G

    def schedule(self, new_state, waiting_queue, scheduled_queue, resources_agenda):
        super().prep_schedule(new_state, waiting_queue, scheduled_queue, resources_agenda)
        super().FCFSNormal()
        self.backfill()

        # execute pre-computed optimal schedule, now its called power manager anw.
        if self.current_time > self.last_event_time:
            if self.current_time in self.power_schedule:
                state_by_id = {node['id']: node for node in new_state}
                newly_allocated_nodes = {node['id'] for node in self.allocated}
                
                for event in self.power_schedule[self.current_time]:
                    nodes_to_action = []
                    
                    if event['type'] == 'switch_on':
                        for node_id in event['nodes']:
                            node = state_by_id.get(node_id)
                            if node and node['state'] == 'sleeping':
                                nodes_to_action.append(node_id)
                        if nodes_to_action:
                            super().push_event(self.current_time, {'type': 'switch_on', 'nodes': nodes_to_action})
                    
                    elif event['type'] == 'switch_off':
                        for node_id in event['nodes']:
                            node = state_by_id.get(node_id)
                            if node and node['state'] == 'active' and node['job_id'] is None and node['id'] not in newly_allocated_nodes:
                                nodes_to_action.append(node_id)
                        if nodes_to_action:
                            super().push_event(self.current_time, {'type': 'switch_off', 'nodes': nodes_to_action})

            self.last_event_time = self.current_time

        return self.events

    def _extract_schedule_from_flow(self, flow_dict, m, n, time_points):
        """Translates a consistent flow dictionary into a power event schedule."""
        power_schedule = {}
        
        # iterate through each time point to build events for that specific time
        for k in range(n):
            t_k = time_points[k]
            nodes_to_power_up = []
            nodes_to_power_down = []
            
            # iterate through each server to see its action at time t_k
            for i in range(m):
                # check for a power-up event at t_k
                if flow_dict.get(f"l_{i},{k}", {}).get(f"u_{i},{k}", 0) > 0.5:
                    nodes_to_power_up.append(i)
                
                # check for a power-down event at t_k
                if flow_dict.get(f"u_{i},{k}", {}).get(f"l_{i},{k}", 0) > 0.5:
                    nodes_to_power_down.append(i)
            
            # add aggregated events to the schedule for the current timestamp
            if nodes_to_power_up:
                power_schedule.setdefault(t_k, []).append({'type': 'switch_on', 'nodes': sorted(list(set(nodes_to_power_up)))})
            if nodes_to_power_down:
                power_schedule.setdefault(t_k, []).append({'type': 'switch_off', 'nodes': sorted(list(set(nodes_to_power_down)))})
        
        return power_schedule
