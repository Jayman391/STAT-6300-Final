import numpy as np
from scipy import stats
from math import floor
import argparse
import matplotlib.pyplot as plt
import os
import csv

class Simulation:

    ## Fixed hyperparameters

    num_timesteps = 100

    initial_users = 20
    initial_groups = 10
    initial_communities = 5

    # group and community preferences
    alpha_group_hyperparameter = 10
    beta_group_hyperparameter = 10 

    alpha_community_hyperparameter = 10
    beta_community_hyperparameter = 10

        
    # Initialize lists to store users and groups
    users = []
    groups = []
    communities = []


    gis = {}
    cis = {}
    uis = {}

    def __init__(self, user_growth_rate, interaction_threshold, new_group_rate, new_community_rate ) -> None:

        self.user_growth_rate = user_growth_rate

        self.interaction_threshold = interaction_threshold

        self.new_group_rate = new_group_rate
        self.new_group_join_chance = new_group_rate / 10

        self.new_community_rate = new_community_rate
        self.new_community_join_chance = new_community_rate / 10

        self.same_community_interaction_ratio = new_community_rate * new_group_rate

    class Community:
        def __init__(self, simulation, group=None):
            self.simulation = simulation 
            # Initialize a community with a list of users and groups
            self.id = len(self.simulation.communities) + 1
            self.groups = [group] if group else []
            self.interactions = []

    class Group:
        def __init__(self, simulation):
            self.simulation = simulation 
            # Initialize a group with an ID and a dictionary tgo track user interactions
            self.id = len(self.simulation.groups) + 1
            self.interactions = {}
            self.community = None

        def join_community(self, community):
            community.groups.append(self)
            self.community = community

    class User:
        def __init__(self, group_alpha, group_beta, community_alpha, community_beta):
            # Initialize a user with ID, group memberships, interaction history, and Beta distribution preferences
            self.id = None

            self.groups = []
            self.communities = []
            self.interaction_history = []

            self.group_preferences = stats.beta(group_alpha, group_beta)
            self.community_preferences = stats.beta(community_alpha, community_beta)
            self.updated_preferences = np.array([1])

        def update_preferences(self):
            # Update user's preferences based on group interactions
            if not self.groups:
                self.updated_preferences = np.array([1])
                return
            else:
                #sort groups by number of interactions
                self.groups.sort(key=lambda group: len(group.interactions))

            total_size = sum([len(group.interactions) for group in self.groups])

            # if size is 0, this must be the first iteration, return uniform
            if total_size == 0:
                self.ccdf = np.array([1])
                return
            else:
                sizes = sorted([len(group.interactions) for group in self.groups])
                self.ccdf = 1 - (np.cumsum(sizes) / total_size)

            group_convolution = np.convolve(self.group_preferences.pdf(np.linspace(0, 1, len(self.groups))), self.ccdf , mode='same')

            self.updated_preferences = np.convolve(group_convolution, self.community_preferences.pdf(np.linspace(0, 1, len(self.groups))), mode='same')

            if np.isnan(self.updated_preferences).any() or np.sum(self.updated_preferences) == 0:
                self.updated_preferences = np.array([1 / len(self.groups)] * len(self.groups))
            else:
                self.updated_preferences /= np.sum(self.updated_preferences)

        def join_group(self, group):
            # Add a group to the user's group list and set initial interactions to 0
            self.groups.append(group)
            group.interactions[self] = 0

        def interact(self, group):
            # Record an interaction with the specified group
            group.interactions[self] = group.interactions.get(self, 0) + 1
            self.interaction_history.append(group.id)

    # Recalculate probabilities at every iteration or after any changes
    def calculate_probabilities(self):
        global community_relative_frequency, group_relative_frequency

        community_relative_frequency = np.array([len(community.groups) for community in self.communities], dtype=float)
        # if community_relative_frequency.sum() != 0:
        community_relative_frequency += 1e-5  # Avoid division by zero
        community_relative_frequency /= community_relative_frequency.sum()

        group_relative_frequency = np.array([sum(group.interactions.values()) for group in self.groups], dtype=float)
        # if group_relative_frequency.sum() != 0:
        group_relative_frequency += 1e-5
        group_relative_frequency /= group_relative_frequency.sum()


    def initialize(self):

        # Initialize users
        for i in range(self.initial_users):
            self.users.append(
                self.User(
                    self.alpha_group_hyperparameter,
                    self.beta_group_hyperparameter,
                    self.alpha_community_hyperparameter,
                    self.beta_community_hyperparameter,
                )
            )
            self.users[-1].id = len(self.users)

        # Initialize communities
        for i in range(self.initial_communities):
            self.communities.append(self.Community(self))

        # Initialize groups
        for i in range(self.initial_groups):
            self.groups.append(self.Group(self))

        # adding the first groups to each community so there is at least one group in each community
        for i in range(len(self.communities)):
            self.groups[i].join_community(self.communities[i])
            # random chance for each user to join the first group of a new community
            for user in self.users:
                if np.random.random() < self.new_community_join_chance:
                    user.join_group(self.groups[i])

        # randomly adding the rest of the groups to communities
        for group in self.groups[len(self.communities):]:
            group.join_community(self.communities[np.random.randint(0, len(self.communities))])
            for user in self.users:
                if np.random.random() < self.new_group_join_chance:
                    user.join_group(group)

        # initialize dictionaries for each group, community, and user
        for group in self.groups:
            self.gis[group.id] = []
        for community in self.communities:
            self.cis[community.id] = []
        for user in self.users:
            self.uis[user.id] = []

        
    def run(self):
        # main loop
        for time in range(self.num_timesteps):
            if time % 10 == 0:
                print(f"Time: {time}")
            # Calculate probabilities
            self.calculate_probabilities()

            # Add new users
            new_users_count = floor(np.random.exponential(self.user_growth_rate))
            for i in range(new_users_count):
                self.users.append(
                    self.User(
                        self.alpha_group_hyperparameter,
                        self.beta_group_hyperparameter,
                        self.alpha_community_hyperparameter,
                        self.beta_community_hyperparameter,
                    )
                )
                self.users[-1].id = len(self.users)

            # Add new groups
            new_groups_count = floor(np.random.exponential(self.new_group_rate))
            for i in range(new_groups_count):
                self.groups.append(self.Group(self))

                # a new community always get made on the first time step
                if time == 0:
                    if new_groups_count == 0:
                        self.groups.append(self.Group(self))
                    self.groups[-1].join_community(self.communities[-1])
                    self.communities[-1].groups.append(self.groups[-1])
                else:
                    # check if the new group forms a new community
                    if np.random.random() < self.new_community_rate:
                        self.communities.append(self.Community(self, self.groups[-1]))
                        self.groups[-1].community = self.communities[-1]
                        # each user has a chance to join the new community
                        for user in self.users:
                            if np.random.random() < self.new_community_join_chance:
                                user.join_group(self.groups[-1])
                        # users[np.random.randint(0, len(users))].join_group(groups[-1])
                    else:
                        # join a random community
                        self.groups[-1].join_community(self.communities[np.random.randint(0, len(self.communities))])

            # Updating dictionaries with new groups, communities, and users
            # and setting their initial values to 0
            for group in self.groups:
                if group.id not in self.gis:
                    self.gis[group.id] = [0]
                self.gis[group.id].append(0)
            for community in self.communities:
                if community.id not in self.cis:
                    self.cis[community.id] = [0]
                self.cis[community.id].append(0)
            for user in self.users:
                if user.id not in self.uis:
                    self.uis[user.id] = [0]
                self.uis[user.id].append(0)

            # Add new users to groups
            for user in self.users:
                self.calculate_probabilities()
                # if there are groups for the user to join that they aren't in
                if len(user.groups) < len(self.groups):
                    # join a group
                    if np.random.random() < self.new_group_join_chance:
                        user.join_group(self.groups[np.random.choice(len(self.groups), p=group_relative_frequency)])

            # Interact with groups
            for user in self.users:
                user.update_preferences()
                interacted_groups = []
                if np.random.uniform() < self.interaction_threshold and user.groups:
                    # print(user.updated_preferences)
                    group = np.random.choice(user.groups, p=user.updated_preferences)
                    user.interact(group)
                    self.gis[group.id][-1] += 1
                    self.cis[group.community.id][-1] += 1
                    self.uis[user.id][-1] += 1

                    # potential bonus interactions within another group in the same community
                    if group.community:
                        while True:
                            if np.random.uniform() < self.same_community_interaction_ratio:
                                community = group.community                    
                                group = np.random.choice(community.groups)
                                user.interact(group)
                                self.gis[group.id][-1] += 1
                                self.cis[group.community.id][-1] += 1
                                self.uis[user.id][-1] += 1
                            else:
                                break

            # Update user preferences
            for user in self.users:
                if user.groups:
                    user.update_preferences()
                    if user.id == 0:
                        print(user.updated_preferences)
                        print(user.group_preferences.pdf(np.linspace(0, 1, len(user.groups))))
                else:
                    user.updated_preferences = np.array([1])

    def plot(self, sim_number):        
        directory_name = f"{self.user_growth_rate}_{self.interaction_threshold}_{self.new_group_rate}_{self.new_community_rate}/{sim_number}"
        os.makedirs(directory_name, exist_ok=True)

        c_sum = []
        c_sum_labels = []
        for i in range(len(self.communities)):
            temp_sum = [0] * self.num_timesteps
            c_vals = np.cumsum(self.cis[i+1])
            # add the values starting from the back
            for j, val in enumerate(reversed(c_vals)):
                temp_sum[-1-j] = val
            c_sum.append(temp_sum)
            c_sum_labels.append(list(self.cis.keys())[i])

        # print the final value for each community
        for c in c_sum_labels[:5]:
            print(c, c_sum[c_sum_labels.index(c)][-1])
            
        # finding the labels for the 5 largest communities
        top_5 = []
        top_5_labels = []
        for i in range(5):
            max_val = 0
            max_index = 0
            for j in range(len(c_sum)):
                if c_sum[j][-1] > max_val and c_sum_labels[j] not in top_5_labels:
                    max_val = c_sum[j][-1]
                    max_index = j
            top_5.append(c_sum[max_index])
            top_5_labels.append(c_sum_labels[max_index])

        for i in range(len(c_sum)):
            if c_sum_labels[i] in top_5_labels:
                plt.plot(c_sum[i][:len(c_sum[i])], label=f"C{i+1}")
            else:
                plt.plot(c_sum[i][:len(c_sum[i])], label=None)

        plt.legend(loc="upper left")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Interactions")
        plt.yscale("log")
        plt.title("Cumulative Interactions of Each Community Over Time")
        plt.savefig(f"{directory_name}/community_interaction_growth.png")
        plt.close()
        
        # Scatter plot for final amount of interactions for each community
        c_sum_final = []
        c_sum_final_labels = []
        for i in range(len(self.communities)):
            c_sum_final.append(c_sum[i][-1])
            c_sum_final_labels.append(list(self.cis.keys())[i])

        plt.scatter(c_sum_final_labels, c_sum_final)
        plt.xlabel("Community")
        plt.ylabel("Final Cumulative Interactions")
        plt.title("Final Cumulative Interactions of Each Community")
        plt.savefig(f"{directory_name}/final_community_interactions.png")
        plt.close()
        

        g_sum = []
        g_sum_labels = []
        for i in range(1, len(self.groups)):
            temp_sum = [0] * self.num_timesteps
            g_vals = np.cumsum(self.gis[i])
            # add the values starting from the back
            for j, val in enumerate(reversed(g_vals)):
                try:
                    temp_sum[j] = val
                except:
                    pass

            temp_sum = temp_sum[::-1]
            g_sum.append(temp_sum)
            g_sum_labels.append(list(self.gis.keys())[i])

        # print the final value for each group
        for g in g_sum_labels[:5]:
            print(g, g_sum[g_sum_labels.index(g)][-1])

        # finding the labels for the 5 largest groups
        top_5 = []
        top_5_labels = []
        for i in range(5):
            max_val = 0
            max_index = 0
            for j in range(len(g_sum)):
                if g_sum[j][-1] > max_val and g_sum_labels[j] not in top_5_labels:
                    max_val = g_sum[j][-1]
                    max_index = j
            top_5.append(g_sum[max_index])
            top_5_labels.append(g_sum_labels[max_index])

        for i in range(len(g_sum)):
            if g_sum_labels[i] in top_5_labels:
                plt.plot(g_sum[i], label=f"G{i+1}")
            else:
                plt.plot(g_sum[i], label=None)

        plt.legend(loc="upper left")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Interactions")
        # plt.yscale("log")
        plt.ylim(bottom=1)
        plt.title("Cumulative Interactions of Each Group Over Time")
        plt.savefig(f"{directory_name}/group_interaction_growth.png")
        plt.close()

        # Scatter plot for final amount of interactions for each group
        g_sum_final = []
        g_sum_final_labels = []
        for i in range(len(g_sum)):
            g_sum_final.append(g_sum[i][-1])
            g_sum_final_labels.append(list(self.gis.keys())[i])

        plt.scatter(g_sum_final_labels, g_sum_final)
        plt.xlabel("Group")
        plt.ylabel("Final Cumulative Interactions")
        plt.title("Final Cumulative Interactions of Each Group")
        plt.savefig(f"{directory_name}/final_group_interactions.png")
        plt.close()

        for u in self.uis:
            self.uis[u] = self.uis[u][:self.num_timesteps]

        # plotting total amount of interactions for each user
        u_sum = []
        u_sum_labels = []
        for i in range(1, len(self.users)):
            temp_sum = [0] * self.num_timesteps
            u_vals = np.cumsum(self.uis[i])
            # add the values starting from the back
            for j, val in enumerate(reversed(u_vals)):
                temp_sum[j] = val

            temp_sum = temp_sum[::-1]
            u_sum.append(temp_sum)
            u_sum_labels.append(list(self.uis.keys())[i])

        # print the final value for each user
        for u in u_sum_labels[:5]:
            print(u, u_sum[u_sum_labels.index(u)][-1])

        # finding the labels for the 5 largest users
        top_5 = []
        top_5_labels = []

        for i in range(5):
            max_val = 0
            max_index = 0
            for j in range(len(u_sum)):
                if u_sum[j][-1] > max_val and u_sum_labels[j] not in top_5_labels:
                    max_val = u_sum[j][-1]
                    max_index = j
            top_5.append(u_sum[max_index])
            top_5_labels.append(u_sum_labels[max_index])

        # Scatter plot for final amount of interactions for each user
        u_sum_final = []
        u_sum_final_labels = []
        for i in range(len(u_sum)):
            u_sum_final.append(u_sum[i][-1])
            u_sum_final_labels.append(list(self.uis.keys())[i])

        plt.scatter(u_sum_final_labels, u_sum_final)
        plt.xlabel("User")
        plt.ylabel("Final Cumulative Interactions")
        plt.title("Cumulative Interactions of Each User")
        plt.savefig(f"{directory_name}/final_user_interactions.png")
        plt.close()

    
    def write_data(self, sim_number):
        directory_name = f"{self.user_growth_rate}_{self.interaction_threshold}_{self.new_group_rate}_{self.new_community_rate}/{sim_number}"
        os.makedirs(directory_name, exist_ok=True)

        # Write User Interactions to CSV
        with open(f"{directory_name}/user_interactions.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            for row in self.uis:
                writer.writerow([row] + self.uis[row])

            file.close()
        
        # Write Group Interactions to CSV
        with open(f"{directory_name}/group_interactions.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            for row in self.gis:
                writer.writerow([row] + self.gis[row])

            file.close()

        # Write Community Interactions to CSV
        with open(f"{directory_name}/community_interactions.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            for row in self.cis:
                writer.writerow([row] + self.cis[row])

            file.close()


        with open(f"{directory_name}/simulation_data.csv", 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['num_users', 'num_groups', 'num_communities', 'num_interactions'])
            writer.writerow([len(self.users), len(self.groups), len(self.communities), sum([len(group.interactions) for group in self.groups])])

            file.close()

        print("Data written to CSV files.")

     
if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Run simulation')
    parser.add_argument('user_growth_rate', type=float, default=1, help='user growth rate')
    parser.add_argument('interaction_threshold', type=float, default=0.5, help='interaction threshold')
    parser.add_argument('new_group_rate', type=float, default=1, help='new group rate')
    parser.add_argument('new_community_rate', type=float, default=1, help='new community rate')
    parser.add_argument('run', type=int, default=1, help='simulation run number')

    args = parser.parse_args()

    sim = Simulation(float(args.user_growth_rate)/10, float(args.interaction_threshold)/10, float(args.new_group_rate)/10, float(args.new_community_rate)/10)
    sim.initialize()
    sim.run()
    #sim.plot(args.run)
    sim.write_data(args.run)
    del sim
    


    