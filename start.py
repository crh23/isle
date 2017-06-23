"""
 * Created by Torsten Heinrich
 */
Translated to python by Davoud Taghawi-Nejad
"""

from __future__ import division
from insurancefirm import InsuranceFirm
from insurancecustomer import InsuranceCustomer
from riskcategory import RiskCategory
from abce import Simulation, gui
from collections import defaultdict
#from 


#import os
import sys
import yaml
import math
import scipy
import scipy.stats
import pdb

simulation_parameters = {'name': 'name',
                         'scheduledEndTime': 100,
                         'numberOfInsurers': 10,
                         'numberOfRiskholders': 1000,
                         'start_cash_insurer': 1000.0,
                         'start_cash_customer': 10000.0,
                         'defaultContractRuntime': 10,
                         'defaultContractExcess': 100,
                         'numberOfRiskCategories': 5,
                         'shareOfCorrelatedRisk': 0.5,
                         'numberOfRiskCategoryDimensions': 2,
                         'riskObliviousSetting': 2,
                         'series': 'testing'#,
                         }

direct_output_suppressed = False
if len(sys.argv) > 1: #exists parameters.yml 
    yamlfilename = sys.argv[1]
    yamlfile = open(yamlfilename, "r")
    spconf = yaml.load(yamlfile)
    simulation_parameters = spconf['simulation_parameters']
    if len(sys.argv) > 2:
        if int(sys.argv[2]) == 1:
            direct_output_suppressed = True
            print("Graphical output will be suppressed")

#@gui(simulation_parameters)
def main(simulation_parameters):
        simulation = Simulation(rounds=simulation_parameters['scheduledEndTime'], processes=1)

        insurancefirms = simulation.build_agents(InsuranceFirm, 'insurancefirm',
                       number=simulation_parameters['numberOfInsurers'],
                       parameters=simulation_parameters)
        insurancecustomers = simulation.build_agents(InsuranceCustomer, 'insurancecustomer',
                       number=simulation_parameters['numberOfRiskholders'],
                       parameters=simulation_parameters)
        allagents = insurancefirms + insurancecustomers
        ic_objects = list(insurancecustomers.do('get_object'))
        if_objects = list(insurancefirms.do('get_object'))
        
        riskcategories = []
        for i in range(simulation_parameters['numberOfRiskCategoryDimensions']):
            riskcategories.append([RiskCategory(0, simulation_parameters['scheduledEndTime']) for i in range(simulation_parameters['numberOfRiskCategories'])])
        
        events = defaultdict(list)
        
        for round in simulation.next_round():
            
            #new_events = insurancecustomers.do('randomAddRisk')
            new_events = []
            
            if round == 0:
                #eventDist = None#scipy.stats.expon(0, 100./1.)
                #eventSizeDist = None#scipy.stats.pareto(2., 0., 10.)
                eventDist = scipy.stats.expon(0, 100./1.)
                eventSizeDist = scipy.stats.pareto(2., 0., 10.)
                bernoulliDistCategory = scipy.stats.bernoulli(simulation_parameters['shareOfCorrelatedRisk']*1./simulation_parameters['numberOfRiskCategoryDimensions'])
                bernoulliDistIndividual = scipy.stats.bernoulli(1-simulation_parameters['shareOfCorrelatedRisk'])
                #workaround (for agent methods with arguments), will not work multi-threaded because of pointer/object reference space mismatch
                new_events = [ic.startAddRisk(15, simulation_parameters['scheduledEndTime'], \
                                                        riskcategories, eventDist, eventSizeDist, bernoulliDistIndividual=bernoulliDistIndividual, bernoulliDistCategory=bernoulliDistCategory) for ic in ic_objects]
                new_events = [event for agent_events in new_events for event in agent_events]
                try:
                    roSetting = simulation_parameters['riskObliviousSetting']           #parameter riskObliviousSetting:
                                                                                        #     if 0: all firms aware of all categories 
                                                                                        #     if 1: all firms unaware of first category, 
                                                                                        #     if 2: half the firms unaware of first category, the other half of the second category
                    #print("DEBUG start read roSetting: Success")
                    if roSetting == 1:
                        [ifirm.set_oblivious(0) for ifirm in if_objects]
                    elif roSetting == 2:
                        assert simulation_parameters['numberOfRiskCategoryDimensions'] > 1
                        noi = simulation_parameters['numberOfInsurers']
                        middle = int(noi/2.)                                            #round does not work as round is redefined as int
                        [ifirm.set_oblivious(0) for ifirm in if_objects[:middle]]
                        [ifirm.set_oblivious(1) for ifirm in if_objects[middle:]]
                except: 
                    #print("DEBUG start read roSetting unsuccessful")
                    #pdb.set_trace()
                    pass
                #pdb.set_trace()
            for risk in events[round]:
                new_events += [risk.explode(round)]		#TODO: does this work with multiprocessing?
            for event_time, risk in new_events:
                if event_time is not None:
                    event_time = math.ceil(event_time)
                    events[event_time].append(risk)
                    assert isinstance(event_time, int)
                    assert risk is not None
                    try:
                        assert event_time >= round
                    except:
                        pdb.set_trace()
            insurancecustomers.do('get_mean_coverage')
            (insurancefirms + insurancecustomers).do('mature_contracts')
            insurancecustomers.do('randomAddCoverage')
            insurancefirms.do('quote')
            insurancecustomers.do('subscribe_coverage')
            insurancefirms.do('add_contract')
            allagents.do('filobl')
            insurancecustomers.do('check_risk')
            #(insurancefirms + insurancecustomers).do('logging')
            (insurancefirms).do('logging')
            #print(sum(list(insurancefirms.do('is_bankrupt'))))
            #print("\nDEBUG start mean cover: ", scipy.mean(insurancecustomers.do('get_mean_coverage')))

        #if not direct_output_suppressed:
        #    simulation.graphs()

if __name__ == '__main__':
    main(simulation_parameters)
