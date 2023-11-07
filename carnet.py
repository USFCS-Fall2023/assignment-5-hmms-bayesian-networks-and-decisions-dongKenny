from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD


def car_queries():
    car_model = BayesianNetwork(
        [
            ("Battery", "Radio"),
            ("Battery", "Ignition"),
            ("Ignition", "Starts"),
            ("Gas", "Starts"),
            ("Starts", "Moves"),
            ("KeyPresent", "Starts")
        ]
    )

    # Defining the parameters using CPT
    cpd_battery = TabularCPD(
        variable="Battery", variable_card=2, values=[[0.70], [0.30]],
        state_names={"Battery": ['Works', "Doesn't work"]},
    )

    cpd_gas = TabularCPD(
        variable="Gas", variable_card=2, values=[[0.40], [0.60]],
        state_names={"Gas": ['Full', "Empty"]},
    )

    cpd_radio = TabularCPD(
        variable="Radio", variable_card=2,
        values=[[0.75, 0.01], [0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Radio": ["turns on", "Doesn't turn on"],
                     "Battery": ['Works', "Doesn't work"]}
    )

    cpd_ignition = TabularCPD(
        variable="Ignition", variable_card=2,
        values=[[0.75, 0.01], [0.25, 0.99]],
        evidence=["Battery"],
        evidence_card=[2],
        state_names={"Ignition": ["Works", "Doesn't work"],
                     "Battery": ['Works', "Doesn't work"]}
    )

    cpd_starts = TabularCPD(
        variable="Starts",
        variable_card=2,
        values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
        evidence=["Ignition", "Gas", "KeyPresent"],
        evidence_card=[2, 2, 2],
        state_names={"Starts": ['yes', 'no'], "Ignition": ["Works", "Doesn't work"], "Gas": ['Full', "Empty"], "KeyPresent": ["yes", "no"]},
    )

    cpd_moves = TabularCPD(
        variable="Moves", variable_card=2,
        values=[[0.8, 0.01], [0.2, 0.99]],
        evidence=["Starts"],
        evidence_card=[2],
        state_names={"Moves": ["yes", "no"],
                     "Starts": ['yes', 'no']}
    )

    cpd_key = TabularCPD(
        variable="KeyPresent", variable_card=2, values=[[0.70], [0.30]],
        state_names={"KeyPresent": ['yes', "no"]},
    )

    # Associating the parameters with the model structure
    car_model.add_cpds(cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_key)

    car_infer = VariableElimination(car_model)

    q_1 = car_infer.query(variables=["Battery"], evidence={"Moves": "yes"})
    q_2 = car_infer.query(variables=["Starts"], evidence={"Radio": "Doesn't turn on"})

    q_3 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works"})
    q_3_2 = car_infer.query(variables=["Radio"], evidence={"Battery": "Works", 'Gas': 'Full'})

    q_4 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no"})
    q_4_2 = car_infer.query(variables=["Ignition"], evidence={"Moves": "no", 'Gas': 'Empty'})

    q_5 = car_infer.query(variables=["Starts"], evidence={"Radio": "turns on", 'Gas': 'Full'})

    q_6 = car_infer.query(variables=["Starts"], evidence={"Ignition": "Works", 'Gas': 'Full', "KeyPresent": "yes"})

    print(q_1)
    print(q_2)

    print(q_3)
    print(q_3_2)

    print(q_4)
    print(q_4_2)

    print(q_5)
    print(q_6)

    print('Probability unchanged for the radio working given the battery is working if the we discover the car has gas')
    print('Probability affected for the ignition failing given the car does not move and we observe there is no gas')
