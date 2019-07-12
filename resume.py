# import common packages
import argparse
import hashlib
import numpy as np
import pickle
import random

# import config file and apply configuration
import isleconfig
# TODO: this whole module could be an argument for start.py


simulation_parameters = isleconfig.simulation_parameters
replic_ID = None
override_no_riskmodels = False

# use argparse to handle command line arguments
parser = argparse.ArgumentParser(description="Model the Insurance sector")
parser.add_argument("--abce", action="store_true", help="[REMOVED] use abce")
parser.add_argument(
    "--oneriskmodel",
    action="store_true",
    help="allow overriding the number of riskmodels from the standard config (with 1)",
)
parser.add_argument(
    "--riskmodels",
    type=int,
    choices=[1, 2, 3, 4],
    help="allow overriding the number of riskmodels from standard config (with 1 or other numbers)",
)
parser.add_argument(
    "--replicid",
    type=int,
    help="if replication ID is given, pass this to the simulation so that the risk profile can be restored",
)
parser.add_argument(
    "--replicating",
    action="store_true",
    help="if this is a simulation run designed to replicate another, override the config file parameter",
)
parser.add_argument(
    "--randomseed", type=float, help="allow setting of numpy random seed"
)
parser.add_argument(
    "--foreground",
    action="store_true",
    help="force foreground runs even if replication ID is given (which defaults to background runs)",
)
parser.add_argument("-p", "--showprogress", action="store_true", help="show timesteps")
parser.add_argument("-v", "--verbose", action="store_true", help="more detailed output")
args = parser.parse_args()

if args.abce:
    raise Exception("ABCE is not and will not be supported")
if args.oneriskmodel:
    isleconfig.oneriskmodel = True
    override_no_riskmodels = 1
if args.riskmodels:
    override_no_riskmodels = args.riskmodels
if args.replicid is not None:
    replic_ID = args.replicid
if args.replicating:
    isleconfig.replicating = True
    assert (
        replic_ID is not None
    ), "Error: Replication requires a replication ID to identify run to be replicated"
if args.randomseed:
    randomseed = args.randomseed
    seed = int(randomseed)
else:
    np.random.seed()
    seed = np.random.randint(0, 2 ** 31 - 1)
if args.foreground:
    isleconfig.force_foreground = True
if args.showprogress:
    isleconfig.showprogress = True
if args.verbose:
    isleconfig.verbose = True

# import isle modules

from insurancefirm import InsuranceFirm
from reinsurancefirm import ReinsuranceFirm


# main function
def main():
    with open("data/simulation_save.pkl", "br") as rfile:
        d = pickle.load(rfile)
        simulation = d["simulation"]
        world = simulation
        np_seed = d["np_seed"]
        random_seed = d["random_seed"]
        time = d["time"]
        simulation_parameters = d["simulation_parameters"]

    insurancefirms_group = list(simulation.insurancefirms)
    reinsurancefirms_group = list(simulation.reinsurancefirms)

    # np.random.seed(seed)
    np.random.set_state(np_seed)
    random.setstate(random_seed)

    # time iteration
    for t in range(time, simulation_parameters["max_time"]):

        # abce time step
        simulation.advance_round(t)

        # create new agents    # TODO: write method for this; this code block is executed almost identically 4 times
        if world.insurance_firm_enters_market(agent_type="InsuranceFirm"):
            parameters = [np.random.choice(world.agent_parameters["insurancefirm"])]
            parameters[0]["id"] = world.get_unique_insurer_id()
            new_insurance_firm = simulation.build_agents(
                InsuranceFirm,
                "insurancefirm",
                parameters=simulation_parameters,
                agent_parameters=parameters,
            )
            insurancefirms_group += new_insurance_firm
            new_insurancefirm_pointer = new_insurance_firm
            world.accept_agents("insurancefirm", new_insurancefirm_pointer, time=t)

        if world.insurance_firm_enters_market(agent_type="ReinsuranceFirm"):
            parameters = [np.random.choice(world.agent_parameters["reinsurancefirm"])]
            parameters[0]["id"] = world.get_unique_reinsurer_id()
            new_reinsurance_firm = simulation.build_agents(
                ReinsuranceFirm,
                "reinsurancefirm",
                parameters=simulation_parameters,
                agent_parameters=parameters,
            )
            reinsurancefirms_group += new_reinsurance_firm
            new_reinsurancefirm_pointer = new_reinsurance_firm
            world.accept_agents("reinsurancefirm", new_reinsurancefirm_pointer, time=t)

        # iterate simulation
        world.iterate(t)

        # log data
        world.save_data()

        if t > 0 and t // 50 == t / 50:
            save_simulation(t, simulation, simulation_parameters, exit_now=False)
        # print("here")

    # finish simulation, write logs
    simulation.finalize()


# save function
def save_simulation(t, sim, sim_param, exit_now=False):
    d = {}
    d["np_seed"] = np.random.get_state()
    d["random_seed"] = random.getstate()
    d["time"] = t
    d["simulation"] = sim
    d["simulation_parameters"] = sim_param
    with open("data/simulation_resave.pkl", "bw") as wfile:
        pickle.dump(d, wfile, protocol=pickle.HIGHEST_PROTOCOL)
    with open("data/simulation_resave.pkl", "br") as rfile:
        file_contents = rfile.read()
    # print("\nSimulation hashes: ", hashlib.sha512(str(d).encode()).hexdigest(), "; ",  hashlib.sha512(str(file_contents).encode()).hexdigest())
    # note that the hash over the dict is for some reason not identical between runs. The hash over the state saved to the file is.
    print(
        "\nSimulation hash: ", hashlib.sha512(str(file_contents).encode()).hexdigest()
    )
    if exit_now:
        exit(0)


# main entry point
if __name__ == "__main__":
    main()
