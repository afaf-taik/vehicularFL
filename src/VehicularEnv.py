# from dataclasses import dataclass
import numpy as np
import random


class V2IChannels:
    def __init__(self, BS_position = None, BS_radius = None, BS_height = None, BS_antenna_gain = None, BS_std_shadow = None, BS_noise_figure = None):
        self.BS_position: float = BS_position if BS_position is not None else np.array([1000.0, 25])  # BS position
        self.BS_radius: float = BS_radius if BS_radius is not None else 2000  # BS cell radius 
        self.BS_height: float = BS_height if BS_height is not None else 25  # BS height
        self.BS_antenna_gain: int = BS_antenna_gain if BS_antenna_gain is not None else 8  # BS antenna gain 3 dBi
        self.BS_std_shadow: int = BS_std_shadow if BS_std_shadow is not None else 8  # Shadowing standard deviation for BS
        self.BS_noise_figure: int = BS_noise_figure if BS_noise_figure is not None else 9  # BS noise figure



class V2VChannels:
    def __init__(self, tx_height = None, rx_height = None, antenna_gain = None, std_shadow = None, noise_figure = None):
        self.tx_height: float = tx_height if tx_height is not None else 1.5  # Tx vehicle height
        self.rx_height: float = rx_height if rx_height is not None else 1.5  # Tx vehicle height
        self.antenna_gain: int = antenna_gain if antenna_gain is not None else 3  # Vehicle antenna gain 3 dBi
        self.std_shadow: int = std_shadow if std_shadow is not None else 3  # Shadowing standard deviation
        self.noise_figure: int = noise_figure if noise_figure is not None else 9  # Vehicle noise figure

class Vehicle:
    def __init__(self, position = None, direction = None, velocity = None, coverage = None, power = None):
        self.position: np.ndarray = position if position is not None else np.array([0.0, 0.0])
        self.direction: str = direction if direction is not None else 'right'  # :right or left
        self.velocity: float = velocity if velocity is not None else 60.0
        self.coverage: float = coverage if coverage is not None else 300.0
        self.power: float = power if power is not None else 1.0


def truncated_normal(mean, stddev, minval, maxval):
    return np.clip(np.random.normal(mean, stddev), minval, maxval)

    # neighbors::Dict{Tuple{Int64,Int64,Int64},Array{Int64,1}} = Dict{Tuple{Int64,Int64,Int64},Array{Int64,1}}()

# x = Vehicle()
# print(x.position[1])
class Freeway(V2VChannels, Vehicle):
    def __init__(self, num_vehicles = None, num_slots = None, num_channels = None):
        self.seed_number: int = 1
        self.rng = random.seed(self.seed_number) #MersenneTwister(1234)
        self.v2v_channels: V2VChannels = V2VChannels()
        self.v2i_channels: V2IChannels = V2IChannels()
        self.num_lanes: int = 6
        #Change to truncnorm 
        self.velocity: np.ndarray = np.zeros(num_vehicles) # 60, 80, 100, 100, 80, 60
        self.lane_width: float = 4.0
        self.freeway_length: float = 2000.0
        self.vitesse : np.ndarray = np.array([19.5,24.5,29.5,29.5,24.5,19.5])
        self.avg_inter_distance: np.ndarray = np.array([0.6944 * v for v in self.vitesse])
        self.forward_lanes: np.ndarray = np.array([2.0, 6.0, 10.0])
        self.backward_lanes: np.ndarray = np.array([14.0, 18.0, 22.0])
        self.lanes: np.ndarray = np.array([2.0, 6.0, 10.0, 22.0, 18.0,  14.0])
        self.vehicles: np.ndarray = [] # array of vehicles objects
        self.txpowers: np.ndarray = [5, 10, 15, 20, 23, 27, 30]  # the power levels
        self.txpower_levels: np.ndarray = range(len(self.txpowers))  # the power levels
        self.sigma: float = 10 ** ((0 - 114 - 30) / 10.0)
        self.num_vehicles: int = num_vehicles if num_vehicles is not None else 1  # number of source vehicles
        # horizon::Float32 = 0.1f0  # simulation time 100 ms.
        # self.bandwidth: float = 10e6
        self.carrier_freq: int = 2  # Carrier frequency
        self.slot_duration: float = 1e-2  # time-slot duration = 10 ms.
        self.num_slots: int = num_slots if num_slots is not None else 1
        self.num_channels: int = num_channels if num_channels is not None else 1  # number of slots. A tperiod is of length 100 ms
        self.max_coverage: float = 300.0
        self.max_power: float = 1.0
        self.coverages: np.ndarray = 300.0 * np.ones(self.num_vehicles) #rand(rng, 500:100.0:1000, numsrcvehicles) #TODO: make it not random
        self.channelgains: np.ndarray = np.zeros((self.num_vehicles, self.num_vehicles+1, self.num_slots, self.num_channels))
        self.pathloss: np.ndarray = np.zeros((self.num_vehicles, self.num_vehicles+1, self.num_slots, self.num_channels))
        self.v2v_channelgains: np.ndarray = np.zeros((self.num_vehicles, self.num_vehicles, self.num_slots, self.num_channels))
        self.v2v_pathloss: np.ndarray = np.zeros((self.num_vehicles, self.num_vehicles, self.num_slots, self.num_channels))
        self.v2v_distances: np.ndarray = np.zeros((self.num_vehicles, self.num_vehicles, self.num_slots, self.num_channels))
        self.v2v_positions: np.ndarray = np.zeros((self.num_slots, self.num_channels, self.num_vehicles, 2))
        self.v2i_channelgains: np.ndarray = np.zeros((self.num_vehicles, self.num_slots, self.num_channels))
        self.v2i_pathloss: np.ndarray = np.zeros((self.num_vehicles, self.num_slots, self.num_channels))
        self.v2i_distances: np.ndarray = np.zeros((self.num_vehicles, self.num_slots, self.num_channels))
        self.bandwidth: float = 1e6  # bandwidth per RB, 1 MHz
        self.move: np.ndarray = np.array([0.2777 * v for v in self.vitesse])
        
    # v2vchannel = V2VChannels()
    def generate_pathloss_shadowing(self, linktype, std_shadow, tx_position, rx_position, tx_height, rx_height, carrier_freq, x):
        ## linktype should be either V2V or V2I
        distance = np.sqrt((tx_position[0] - rx_position[0])**2 + (tx_position[1] - rx_position[1])**2)
        if linktype == 'V2V':
            d_bp = 4 * (tx_height - 1) * (rx_height - 1) * carrier_freq * 10**9 / (3 * 10**8)
            A = 22.7
            B = 41.0
            C = 20
            if distance <= 3:
                PL = A * np.log10(3) + B + C * np.log10(carrier_freq / 5)
            elif distance <= d_bp:
                PL = A * np.log10(distance) + B + C * np.log10(carrier_freq / 5)
            else:
                PL = 40 * np.log10(distance) + 9.45 - 17.3 * np.log10((tx_height - 1) * (rx_height - 1)) + 2.7 * np.log10(carrier_freq / 5)
        elif linktype == 'V2I':
            PL = 128.1 + 37.6 * np.log10(np.sqrt((tx_height - rx_height)**2 + distance**2) / 1000);
        else:
            ValueError('linktype takes only two values: V2V or V2I')
        # x = randn(rng)
        shadowing_pathloss = x * std_shadow + PL
        return shadowing_pathloss, distance

    add_vehicle = lambda self, veh: self.vehicles.append(veh)

    def generate_vehicles(self):
        ##
        veh_positions = [] # array of tuples
        idx_vehicles = []
        flag = False
        # lanes_mapping = Dict{key : value} where key = lane number and value is the y-coordinate of the lane
        lanes_mapping = {i : 2.0 * (2 * i + 1) for i in range(self.num_lanes)}
        key_list = list(lanes_mapping.keys())
        val_list = list(lanes_mapping.values())                
        # generate all vehicle positions and store in veh_positions
        for ith_lane in range(self.num_lanes):
            # for each lane, generate random points according to a Poisson distribution,
            # then generate their x- and y-coordinates and store the values in veh_positions
            num_points = np.random.poisson(self.freeway_length / self.avg_inter_distance[ith_lane])
            horz_position = np.random.uniform(low = 0, high = self.freeway_length + 1e-1, size = num_points)
            # println(horzposition)
            vert_position = (self.lane_width * (ith_lane) + self.lane_width / 2) * np.ones(len(horz_position))
            # println(vertposition)
            positions = [z for z in zip(horz_position, vert_position)]
            veh_positions.append(positions)


        vehicle_positions = np.concatenate(veh_positions) # transform a list of lists of tuples to list of lists
        # vehicle_positions = [collect(x) for x in vehicle_positions]
        num_vehicles = np.shape(vehicle_positions)[0] # all vehicles generated. 
        # We will take only the needed number, i.e., self.num_vehicles
        if num_vehicles < self.num_vehicles:
            flag = True  # the generated vehicles are not enough
            return flag, vehicle_positions, idx_vehicles
        # generate V and W positions
        permuted_idx = np.random.permutation(num_vehicles)
        idx_vehicles = permuted_idx[0:self.num_vehicles]  # randomly pick src_veh
        # idx_vehicles = idx_vehicles[:]

        cv = 0 # coverage index
        for iv in idx_vehicles:
            if vehicle_positions[iv][1] in self.forward_lanes:
                dict_position = val_list.index(vehicle_positions[iv][1])
                lane = key_list[dict_position]
                vehicle = Vehicle(
                    position = vehicle_positions[iv],
                    direction = 'right',
                    velocity = self.vitesse[lane],
                    coverage = self.coverages[cv],
                )
                self.add_vehicle(vehicle)
            else:
                dict_position = val_list.index(vehicle_positions[iv][1])
                lane = key_list[dict_position]
                vehicle = Vehicle(
                    position = vehicle_positions[iv],
                    direction = 'left',
                    velocity = self.vitesse[lane],
                    coverage = self.coverages[cv],
                )
                self.add_vehicle(vehicle)
            cv += 1 # coverage index
        return flag, vehicle_positions, idx_vehicles

    def generate_gain_mobility(self):
        ##
        channel_gains = np.zeros((self.num_vehicles, self.num_vehicles+1, self.num_slots, self.num_channels))
        shadowing_pathloss_dB = np.zeros((self.num_vehicles, self.num_vehicles+1, self.num_slots, self.num_channels))
        vehicle_positions = np.zeros((self.num_slots, self.num_channels, self.num_vehicles, 2))
        for v in range(self.num_vehicles):
            for xy in range(2): # xy ∈ xy_coord
                vehicle_positions[0, 0, v, xy] = self.vehicles[v].position[xy] # initial position
        # xshadow = randn(self.rng)
        for t in range(self.num_channels):
            for s in range(self.num_slots):
                for v in range(self.num_vehicles):
                    for w in range(self.num_vehicles+1): # +1 for the BS
                        if v != w:
                            if w < self.num_vehicles:
                                shadowing_pathloss_dB[v, w, s, t], self.v2v_distances[v, w, s, t] = self.generate_pathloss_shadowing('V2V', self.v2v_channels.std_shadow, self.vehicles[v].position, self.vehicles[w].position, self.v2v_channels.tx_height, self.v2v_channels.rx_height, self.carrier_freq, np.random.normal(loc=0, scale=1))
                            else:
                                shadowing_pathloss_dB[v, w, s, t], self.v2i_distances[v, s, t] = self.generate_pathloss_shadowing('V2I', self.v2i_channels.BS_std_shadow, self.vehicles[v].position, self.v2i_channels.BS_position, self.v2v_channels.tx_height, self.v2i_channels.BS_height, self.carrier_freq, np.random.normal(loc=0, scale=1))
                fast_fading_dB = shadowing_pathloss_dB[:, :, s, t] - 20 * np.log10(np.abs(np.random.normal(loc=0, scale=1, size=(self.num_vehicles, self.num_vehicles+1)) + 1j * np.random.normal(loc=0, scale=1, size=(self.num_vehicles, self.num_vehicles+1))) / np.sqrt(2))
                channel_gains[:, :-1, s, t] = 10 ** ((-fast_fading_dB[:, :-1] + (2 * self.v2v_channels.antenna_gain - self.v2v_channels.noise_figure)) / 10)
                channel_gains[:, -1, s, t] = 10 ** ((-fast_fading_dB[:, -1] + (self.v2v_channels.antenna_gain + self.v2i_channels.BS_antenna_gain - self.v2i_channels.BS_noise_figure)) / 10)

                self.move_vehicles()
                # println(self.srcvehicles[1].position)
                if s < self.num_slots - 1:
                    for v in range(self.num_vehicles):
                        for xy in range(2):
                            vehicle_positions[s+1, t, v, xy] = self.vehicles[v].position[xy]
                elif s == self.num_slots - 1 and t < self.num_channels - 1:
                    for v in range(self.num_vehicles):
                        for xy in range(2):
                            vehicle_positions[0, t+1, v, xy] = self.vehicles[v].position[xy]

        for v in range(self.num_vehicles):
            channel_gains[v, v, :, :] = 0
        self.channelgains = np.divide(channel_gains, self.sigma)
        self.pathloss = 10 ** (-shadowing_pathloss_dB / 10)
        for v in range(self.num_vehicles):
            self.pathloss[v, v, :, :] = 0
        return vehicle_positions            


    def generate_velocities(self):
     #Generate velocities of each vehicle depending on which lane they are

    #[17,22],[22,27],[27,33]
        means = [19.5,24.5,29.5]
        stddev = 1.75
        minvals = [17,22,27]
        maxvals = [22,27,33]
        direction = [1,1]
        velo = np.zeros(self.num_vehicles)
        pies = []
        for i in range(self.num_vehicles):
            p = np.argwhere(self.vehicles[i].position[1] == self.lanes)[0]%3
            pies.append(p)
            direct = int(bool(self.vehicles[i].position[1] in self.forward_lanes))
            velo[i] =  truncated_normal(means[p[0]], stddev, minvals[p[0]], maxvals[p[0]])*direction[direct]
        #print(velo) 
        #print(pies)                   
        return velo

    def rate_stay(self):
    #Predict T_k  of each vehicle depending on its location and speed
        Tk = np.zeros(self.num_vehicles)
        for i in range(self.num_vehicles):
            Tk[i]= (self.freeway_length - self.vehicles[i].position[0])/self.velocity[i]
        return Tk

    def LLT(self, TR = 300, eps = 1e-20):
    # """Calculate link lifetime
    #  should be changed to only use cluster_head_indexes instead of everyone
    #     TR : Transmission Range = 300 m 
    #     eps is used to avoid division by 0 in case speed_i = speed_j
    #     We still need to find a solution for the case of v_i = v_j, because even with eps, LLT = 0 does not make sense"""
        LLT_ = np.zeros((self.num_vehicles,self.num_vehicles))
        direction = [-1,1]        
        for i in range(self.num_vehicles):
            directi = int(bool(self.vehicles[i].position[1] in self.forward_lanes))            
            for j in range(self.num_vehicles):
                if(i==j):
                    continue
                directj = int(bool(self.vehicles[j].position[1] in self.forward_lanes))              
                deltaa = abs(self.velocity[i]-self.velocity[j])
                product = direction[directi] * direction[directj]
                LLT_[i][j] = (deltaa * product * abs(self.vehicles[i].position[0]-self.vehicles[j].position[0]) + deltaa * TR)/ max(deltaa**2,eps)
                if(LLT_[i][j]<0):
                    print('directions', product)
        return LLT_    

    def move_vehicles(self):
        lanes_mapping = {i : 2.0 * (2*i + 1) for i in range(self.num_lanes)}
        key_list = list(lanes_mapping.keys())
        val_list = list(lanes_mapping.values())
        for v in range(self.num_vehicles):
            v_lane_y = self.vehicles[v].position[1]
            dict_position = val_list.index(v_lane_y)
            v_lane = key_list[dict_position]
            if v_lane_y in self.forward_lanes:
                # move forward (right)
                δ = self.vehicles[v].position[0] + self.move[v_lane]
                if δ < self.freeway_length:
                    self.vehicles[v].position[0] = δ
                else:
                    self.vehicles[v].position[0] = 0
            else:
                # move backward (left)
                δ = self.vehicles[v].position[0] - self.move[v_lane]
                if δ > 0:
                    self.vehicles[v].position[0] = δ
                else:
                    self.vehicles[v].position[0] = self.freeway_length
    def build_environment(self):
        ### Generate the vehciles
        flag = True
        # veh_pos = zeros(Float64)
        # ind_src_veh = zeros(Int64)
        # ind_dst_veh = zeros(Int64)
        while True:
            # Generate traffic on the highway
            flag = self.generate_vehicles()[0]
            print('i am here')
            if flag == False:
                break
        ### Generate the channel gain (following the mobility model)
        # num_slots = T
        # veh_pos_final = veh_pos
        self.velocity = self.generate_velocities()
        vehicle_positions = self.generate_gain_mobility()
        ### Plot the positions in each slot
        # for v in range(self.num_vehicles):
        #     for t in range(self.num_channels):
        #         for s in range(self.num_slots):
        #             w_veh = []
        #             for w in range(self.num_vehicles):
        #                 if v != w:
        #                     if (vehicle_positions[s, t, v, 0] - vehicle_positions[s, t, w, 0])**2 / self.vehicles[v].coverage**2 + (vehicle_positions[s, t, v, 1] - vehicle_positions[s, t, w, 1])**2 / self.vehicles[w].coverage**2 <= 1:
        #                         w_veh.append(w)
        #             # neighbors[v, s] = wveh
        #             self.vehicles[v].neighbors[v, s, t] = w_veh
        self.v2v_positions = vehicle_positions


    
# def LLT(x,y,speeds, cluster_heads_indexes, TR = 300, eps = 1e-08):
#   """Calculate link lifetime
#   should be changed to only use cluster_head_indexes instead of everyone
#     TR : Transmission Range = 300 m 
#     eps is used to avoid division by 0 in case speed_i = speed_j
#     We still need to find a solution for the case of v_i = v_j, because even with eps, LLT = 0 does not make sense"""
#     LLT_ = np.zeros(len(x),len(x))
#   for i in range len(x):
#       for j in range(i,len(x)):
#           deltaa = speeds[i]-speeds[j]
#           LLT_[i][j] = (-deltaa * (x[i]-x[j]) + deltaa * TR)/ max(deltaa**2,eps)
#   return LLT_

# def filter_LLT(LLT_, modelsize, Ttrain):
#   '''Compare LLT with train + upload time
#   '''
#   return 
if __name__=='__main__':
    env = Freeway(num_vehicles = 3, num_slots = 1, num_channels = 2)
    env.build_environment()
    print("channel gains")
    print(env.channelgains)
    print('velocities',env.velocity)
    LLT_ = env.LLT()
    print(LLT_)