#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-

import copy
import numpy as np
from scipy import stats
import datetime


def example_3():
    """
    :return: nf_list_dict (cumulative firing counts of transitions)
    :adjustable parameters: total_time, min_tau, L_info, mP0, mP3, mP9, lambda_fun_dict, pP1T2, pP1T1, pP1T8, pP2T4, pP2T10
                            immediate_transition_list, deterministic_distribution_transition_dict
    : notation: All parameters are non-negative.
    """

    total_time = 80  # total iteration time
    min_tau = 0.1  # minimum time for iteration (τ)
    # required time interval (others default to inf)
    L_info = {"T6": [650, 200, 1],
              "T13": [1000, 1000, 1],
              "T14": [1, 1000, 1000]}

    # initial marking (others default to 0)
    mP0 = 10000
    mP3 = 8000
    mP9 = 300
    # time list
    t_list = np.arange(0, total_time, min_tau)

    lambda_fun_dict = dict()  # firing rate function  (λ(t))
    lambda_fun_dict["T0"] = lambda t: stats.lognorm.pdf(t, 1.2, loc=0, scale=np.exp(2.2)) / \
                                      (1 - stats.lognorm.cdf(t, 1.2, loc=0, scale=np.exp(2.2)))
    lambda_fun_dict["T1"] = lambda t: stats.lognorm.pdf(t, 0.8, loc=0, scale=np.exp(2)) / \
                                      (1 - stats.lognorm.cdf(t, 0.8, loc=0, scale=np.exp(2)))
    lambda_fun_dict["T2"] = lambda t: np.zeros_like(t)
    lambda_fun_dict["T3"] = lambda t: 1 / 5 * np.ones_like(t)
    lambda_fun_dict["T4"] = lambda t: np.zeros_like(t)
    lambda_fun_dict["T5"] = lambda t: 2.5 / 30 * (t / 30) ** (2.5 - 1)
    lambda_fun_dict["T6"] = lambda t: 2 / 5 * (t / 5) ** (2 - 1)
    lambda_fun_dict["T7"] = lambda t: np.zeros_like(t)
    lambda_fun_dict["T8"] = lambda t: np.zeros_like(t)
    lambda_fun_dict["T9"] = lambda t: np.zeros_like(t)
    lambda_fun_dict["T10"] = lambda t: np.zeros_like(t)
    lambda_fun_dict["T11"] = lambda t: 2.5 / 30 * (t / 30) ** (2.5 - 1)
    lambda_fun_dict["T12"] = lambda t: 1.5 / 25 * (t / 25) ** (1.5 - 1)
    lambda_fun_dict["T13"] = lambda t: 1.6 / 2.5 * (t / 2.5) ** (1.6 - 1)
    lambda_fun_dict["T15"] = lambda t: np.zeros_like(t)
    lambda_fun_dict["T16"] = lambda t: np.zeros_like(t)
    lambda_fun_dict["T14"] = lambda t: 1.6 / 2.5 * (t / 2.5) ** (1.6 - 1)

    # per-selection function
    pP1T2 = 0.88
    pP1T1 = 0.06
    pP1T8 = 0.06
    pP2T4 = 0.8
    pP2T10 = 0.2

    # immediate transition
    immediate_transition_list = ["T2", "T4", "T7", "T8", "T10"]
    # deterministic distribution
    deterministic_distribution_transition_dict = {"T9": 7}

    """
    It is strongly recommended not to change the parameters below, otherwise, uncertain errors may occur.
    """

    cumsum_lambda_dict = dict()  # cumulative firing rates
    delta_lambda_dict = dict()  # firing rates

    # mean function of the enabling count(The enabling conditions of different transitions have been considered)
    # "T6", "T13" and "T14" used equation(29), while others used equation(24).
    delta_nd_lists_dict = {"T6": [],
                           "T7": np.zeros((len(t_list), len(t_list) + 1), dtype=float),
                           "T5": np.zeros((len(t_list), len(t_list) + 1), dtype=float),
                           "T11": np.zeros((len(t_list), len(t_list) + 1), dtype=float),
                           "T13": [],
                           "T15": np.zeros((len(t_list), len(t_list) + 1), dtype=float),
                           "T12": np.zeros((len(t_list), len(t_list) + 1), dtype=float),
                           "T14": [],
                           "T16": np.zeros((len(t_list), len(t_list) + 1), dtype=float),
                           }
    for transition_name in L_info.keys():
        delta_nd_lists_dict[transition_name].append(np.zeros((0, 0, 0), dtype=float))
        delta_nd_lists_dict[transition_name].append(np.zeros((0, 0), dtype=float))
        delta_nd_lists_dict[transition_name].append(np.zeros((0, 0), dtype=float))
        delta_nd_lists_dict[transition_name].append(np.zeros((0), dtype=float))
    # mean function of the cumulative enabling count
    nd_list_dict = {}
    # mean function of the cumulative firing count
    nf_list_dict = {}
    # mean function of the enabling count
    delta_nd_list_dict = {}
    # mean function of the firing count
    delta_nf_list_dict = {}
    # mean function of the provided token count
    place_num_list_dict = {'P4': np.zeros_like(np.concatenate(([0, 0], t_list))),
                           'P5': np.zeros_like(np.concatenate(([0, 0], t_list))),
                           'P6': np.zeros_like(np.concatenate(([0, 0], t_list))),
                           'P7': np.zeros_like(np.concatenate(([0, 0], t_list))),
                           'P8': np.zeros_like(np.concatenate(([0, 0], t_list))),
                           'P10': np.zeros_like(np.concatenate(([0, 0], t_list))),
                           'P11': np.zeros_like(np.concatenate(([0, 0], t_list))),
                           }
    # init parameter
    for transition_name in lambda_fun_dict.keys():
        nf_list_dict[transition_name] = np.zeros_like(t_list)
        nf_list_dict[transition_name] = np.concatenate(([0], nf_list_dict[transition_name]))
        nd_list_dict[transition_name] = np.zeros_like(t_list)
        nd_list_dict[transition_name] = np.concatenate(([0, 0], nd_list_dict[transition_name]))
        delta_nd_list_dict[transition_name] = np.zeros_like(t_list)
        delta_nf_list_dict[transition_name] = np.zeros_like(t_list)

        delta_lambda_dict[transition_name] = lambda_fun_dict[transition_name](t_list) * min_tau
        if transition_name in immediate_transition_list:  # immediate transitions
            delta_lambda_dict[transition_name][:] = 100
            delta_lambda_dict[transition_name][0] = 1
        if transition_name in deterministic_distribution_transition_dict.keys():  # deterministic distribution
            delta_lambda_dict[transition_name][int(deterministic_distribution_transition_dict[transition_name] // min_tau):] = 100
            delta_lambda_dict[transition_name][int(deterministic_distribution_transition_dict[transition_name] // min_tau)] = 1

        cumsum_lambda_dict[transition_name] = np.cumsum(delta_lambda_dict[transition_name])
        if transition_name in immediate_transition_list:
            cumsum_lambda_dict[transition_name][:] = 100
        if transition_name in deterministic_distribution_transition_dict.keys():
            cumsum_lambda_dict[transition_name][int(deterministic_distribution_transition_dict[transition_name] // min_tau):] = 100
        cumsum_lambda_dict[transition_name] = np.concatenate(([0], cumsum_lambda_dict[transition_name]))

    nd_list_dict['T0'][1] = mP0
    nd_list_dict['T3'][1] = mP3
    nd_list_dict['T9'][1] = mP9

    # recursive probabilities
    x_j = {
        'T6': {'P4': np.zeros_like(np.concatenate(([0], t_list))), 'P5': np.zeros_like(np.concatenate(([0], t_list)))},
        'T7': {'P5': np.zeros_like(np.concatenate(([0], t_list)))},
        'T5': {'P4': np.zeros_like(np.concatenate(([0], t_list)))},
        'T11': {'P10': np.zeros_like(np.concatenate(([0], t_list)))},
        'T13': {'P10': np.zeros_like(np.concatenate(([0], t_list))), 'P7': np.zeros_like(np.concatenate(([0], t_list)))},
        'T15': {'P7': np.zeros_like(np.concatenate(([0], t_list)))},
        'T12': {'P11': np.zeros_like(np.concatenate(([0], t_list)))},
        'T14': {'P8': np.zeros_like(np.concatenate(([0], t_list))), 'P11': np.zeros_like(np.concatenate(([0], t_list)))},
        'T16': {'P8': np.zeros_like(np.concatenate(([0], t_list)))},
    }
    y_j = copy.deepcopy(x_j)
    # mean function of the cumulative returned token count
    return_nd = {'P4': np.zeros_like(t_list), 'P5': np.zeros_like(t_list), 'P6': np.zeros_like(t_list),
                 'P7': np.zeros_like(t_list), 'P8': np.zeros_like(t_list), 'P10': np.zeros_like(t_list), 'P11': np.zeros_like(t_list)}

    # classification probability of enabling instant (1 - Ψ)
    psi = {'T6': {'P4': 0, 'P5': 0},
           'T7': {'P5': 0},
           'T5': {'P4': 0},
           'T11': {'P10': 0},
           'T13': {'P10': 0, 'P7': 0},
           'T15': {'P7': 0},
           'T12': {'P11': 0},
           'T14': {'P8': 0, 'P11': 0},
           'T16': {'P8': 0},
           }
    # classification probability of arriving instant (1 - Φ)
    phi = copy.deepcopy(psi)
    # structural information
    struct_info1 = {'T6': ['P4', 'P5'],
                    'T7': ['P5'],
                    'T5': ['P4'],
                    'T11': ['P10'],
                    'T13': ['P10', 'P7'],
                    'T15': ['P7'],
                    'T12': ['P11'],
                    'T14': ['P8', 'P11'],
                    'T16': ['P8'],
                    }
    struct_info2 = {'T6': ['T5', 'T7'],
                    'T7': ['T6'],
                    'T5': ['T6'],
                    'T11': ['T13'],
                    'T13': ['T11', 'T15'],
                    'T15': ['T13'],
                    'T12': ['T14'],
                    'T14': ['T16', 'T12'],
                    'T16': ['T14'],
                    }
    first_time = datetime.datetime.now()
    print('Methods 1 and 2 started!')
    record_time_list = []
    for t_list_index, _ in enumerate(t_list):
        print('current_time:{:.3f}'.format((t_list_index + 1) * min_tau),
              '({:.2f}%)'.format((t_list_index + 1) * min_tau/total_time * 100))
        # delta_nd
        for transition_name in lambda_fun_dict.keys():
            delta_nd_list_dict[transition_name][t_list_index] = \
                nd_list_dict[transition_name][t_list_index + 1] - nd_list_dict[transition_name][t_list_index]

        # classification probability of enabling instant (1 - Ψ)
        for transition_name in psi.keys():
            for place_name in psi[transition_name].keys():
                psi[transition_name][place_name] = \
                    delta_nd_list_dict[transition_name][t_list_index] / \
                    (place_num_list_dict[place_name][t_list_index + 1] - nd_list_dict[transition_name][t_list_index] + 1e-300)
        # classification probability of arriving instant (1 - Φ)
        for transition_name in psi.keys():
            for place_name in psi[transition_name].keys():
                phi[transition_name][place_name] = \
                    (place_num_list_dict[place_name][t_list_index + 1] - place_num_list_dict[place_name][t_list_index]) / \
                    (place_num_list_dict[place_name][t_list_index + 1] - nd_list_dict[transition_name][t_list_index] + 1e-300)

        for transition_name in struct_info1.keys():
            for place_index in range(len(struct_info1[transition_name])):
                # x_j
                place_name = struct_info1[transition_name][place_index]
                other_transition_name = struct_info2[transition_name][place_index]
                if 1 - psi[transition_name][place_name] != 1:
                    # equation (52)
                    x_j[transition_name][place_name][1:t_list_index + 1] = y_j[transition_name][place_name][1:t_list_index + 1] * \
                                                                           (1 - phi[transition_name][place_name])
                    """
                        x_j[transition_name][place_name][t_list_index + 1] + x_j[transition_name][place_name][0] = 1 - x_j[transition_name][place_name][1:t_list_index + 1].sum()
                        x_j[transition_name][place_name][t_list_index + 1] / x_j[transition_name][place_name][0] = (1 - psi[other_transition_name][place_name]) / psi[other_transition_name][place_name]
                    """
                    x_j[transition_name][place_name][t_list_index + 1] = (1 - x_j[transition_name][place_name][1:t_list_index + 1].sum()) * \
                                                                         psi[other_transition_name][place_name]
                    x_j[transition_name][place_name][0] = (1 - x_j[transition_name][place_name][1:t_list_index + 1].sum()) * \
                                                          (1 - psi[other_transition_name][place_name])
                else:
                    # equation (54)
                    x_j[transition_name][place_name][0:t_list_index + 2] = 0
                    x_j[transition_name][place_name][t_list_index + 1] = 1
                # y_j
                if 1 - psi[transition_name][place_name] != 0:
                    # equation (55)
                    y_j[transition_name][place_name][1:t_list_index + 1] = y_j[transition_name][place_name][1:t_list_index + 1] * \
                                                                           (1 - phi[transition_name][place_name])
                    """
                    y_j[transition_name][place_name][t_list_index + 1] + y_j[transition_name][place_name][0] = 1 - y_j[transition_name][place_name][1:t_list_index + 1].sum()
                    y_j[transition_name][place_name][t_list_index + 1] / y_j[transition_name][place_name][0] = (1 - psi[other_transition_name][place_name]) / psi[other_transition_name][place_name]
                    """
                    y_j[transition_name][place_name][t_list_index + 1] = (1 - y_j[transition_name][place_name][1:t_list_index + 1].sum()) * \
                                                                         psi[other_transition_name][place_name]
                    y_j[transition_name][place_name][0] = (1 - y_j[transition_name][place_name][1:t_list_index + 1].sum()) * \
                                                          (1 - psi[other_transition_name][place_name])
                else:
                    # equation (57)
                    y_j[transition_name][place_name][0:t_list_index + 2] = 0
                    y_j[transition_name][place_name][0] = 1

        # state transfer probability (equations (64) and (68))
        for transition_name in struct_info1.keys():
            if len(struct_info1[transition_name]) == 1:
                place_name = struct_info1[transition_name][0]
                other_transition_name = struct_info2[transition_name][0]
                delta_nd_lists_dict[transition_name][:t_list_index, t_list_index + 1] = delta_nd_lists_dict[transition_name][:t_list_index, 0] * \
                                                                                        psi[other_transition_name][place_name]
                delta_nd_lists_dict[transition_name][:t_list_index, 0] *= (1 - psi[other_transition_name][place_name])
            if len(struct_info1[transition_name]) == 2:
                place_name0 = struct_info1[transition_name][0]
                other_transition_name0 = struct_info2[transition_name][0]
                place_name1 = struct_info1[transition_name][1]
                other_transition_name1 = struct_info2[transition_name][1]
                new_delta_nd_lists1 = delta_nd_lists_dict[transition_name][2] * psi[other_transition_name0][place_name0]
                delta_nd_lists_dict[transition_name][2] *= (1 - psi[other_transition_name0][place_name0])
                new_delta_nd_lists2 = delta_nd_lists_dict[transition_name][1] * psi[other_transition_name1][place_name1]
                delta_nd_lists_dict[transition_name][1] *= (1 - psi[other_transition_name1][place_name1])

                new_delta_nd_lists3 = delta_nd_lists_dict[transition_name][3] * psi[other_transition_name0][place_name0] * psi[other_transition_name1][place_name1]
                new_delta_nd_lists4 = delta_nd_lists_dict[transition_name][3] * psi[other_transition_name0][place_name0] * (1 - psi[other_transition_name1][place_name1])
                new_delta_nd_lists5 = delta_nd_lists_dict[transition_name][3] * (1 - psi[other_transition_name0][place_name0]) * psi[other_transition_name1][place_name1]

                delta_nd_lists_dict[transition_name][3] *= (1 - psi[other_transition_name0][place_name0]) * (1 - psi[other_transition_name1][place_name1])

                delta_nd_lists_dict[transition_name][0] = np.concatenate((delta_nd_lists_dict[transition_name][0],
                                                                          np.expand_dims(new_delta_nd_lists1, axis=1)), axis=1)
                new_delta_nd_lists6 = np.concatenate((new_delta_nd_lists2, np.expand_dims(new_delta_nd_lists3, axis=1)), axis=1)
                delta_nd_lists_dict[transition_name][0] = np.concatenate((delta_nd_lists_dict[transition_name][0],
                                                                          np.expand_dims(new_delta_nd_lists6, axis=2)), axis=2)
                delta_nd_lists_dict[transition_name][1] = np.concatenate((delta_nd_lists_dict[transition_name][1],
                                                                          np.expand_dims(new_delta_nd_lists4, axis=1)), axis=1)
                delta_nd_lists_dict[transition_name][2] = np.concatenate((delta_nd_lists_dict[transition_name][2],
                                                                          np.expand_dims(new_delta_nd_lists5, axis=1)), axis=1)


        # equation (67)
        for transition_name in struct_info1.keys():
            if len(struct_info1[transition_name]) == 1:
                place_name = struct_info1[transition_name][0]
                delta_nd_lists_dict[transition_name][t_list_index, 0:t_list_index + 2] = x_j[transition_name][place_name][0:t_list_index + 2] * \
                                                                                         delta_nd_list_dict[transition_name][t_list_index]
            if len(struct_info1[transition_name]) == 2:
                place_name0 = struct_info1[transition_name][0]
                place_name1 = struct_info1[transition_name][1]

                new_delta_nd_lists = \
                    x_j[transition_name][place_name0][0:t_list_index + 2].reshape((-1, 1)) * \
                    x_j[transition_name][place_name1][0:t_list_index + 2].reshape((1, -1)) * \
                    delta_nd_list_dict[transition_name][t_list_index]
                delta_nd_lists_dict[transition_name][0] = np.concatenate((delta_nd_lists_dict[transition_name][0],
                                                                          np.expand_dims(new_delta_nd_lists[
                                                                          max(t_list_index - L_info[transition_name][0] + 1, 1):t_list_index + 2,
                                                                          max(t_list_index - L_info[transition_name][2] + 1, 1):t_list_index + 2], axis=0)),
                                                                         axis=0)
                delta_nd_lists_dict[transition_name][1] = np.concatenate((delta_nd_lists_dict[transition_name][1],
                                                                          np.expand_dims(new_delta_nd_lists[
                                                                          max(t_list_index - L_info[transition_name][0] + 1, 1):t_list_index + 2,
                                                                          0], axis=0)),
                                                                         axis=0)
                delta_nd_lists_dict[transition_name][2] = np.concatenate((delta_nd_lists_dict[transition_name][2],
                                                                          np.expand_dims(new_delta_nd_lists[
                                                                          0,
                                                                          max(t_list_index - L_info[transition_name][2] + 1, 1):t_list_index + 2], axis=0)),
                                                                         axis=0)
                delta_nd_lists_dict[transition_name][3] = np.concatenate((delta_nd_lists_dict[transition_name][3],
                                                                          np.expand_dims(new_delta_nd_lists[0, 0], axis=0)),
                                                                         axis=0)
                # the tokens that can be safely discarded
                if t_list_index >= L_info[transition_name][1]:
                    delta_nd_lists_dict[transition_name][0] = np.delete(delta_nd_lists_dict[transition_name][0], 0, axis=0)
                    delta_nd_lists_dict[transition_name][1] = np.delete(delta_nd_lists_dict[transition_name][1], 0, axis=0)
                    delta_nd_lists_dict[transition_name][2] = np.delete(delta_nd_lists_dict[transition_name][2], 0, axis=0)
                    delta_nd_lists_dict[transition_name][3] = np.delete(delta_nd_lists_dict[transition_name][3], 0, axis=0)
                if t_list_index >= L_info[transition_name][0]:
                    delta_nd_lists_dict[transition_name][0] = np.delete(delta_nd_lists_dict[transition_name][0], 0, axis=1)
                    delta_nd_lists_dict[transition_name][1] = np.delete(delta_nd_lists_dict[transition_name][1], 0, axis=1)
                if t_list_index >= L_info[transition_name][2]:
                    delta_nd_lists_dict[transition_name][0] = np.delete(delta_nd_lists_dict[transition_name][0], 0, axis=2)
                    delta_nd_lists_dict[transition_name][2] = np.delete(delta_nd_lists_dict[transition_name][2], 0, axis=1)

        # mean function of the firing count
        for transition_name in ["T0", "T1", "T3", "T9", "T2", "T4", "T8", "T10"]:
            delta_nf_list_dict[transition_name][t_list_index] = (delta_nd_list_dict[transition_name][:t_list_index + 1] *
                                                                 np.exp(-cumsum_lambda_dict[transition_name][t_list_index::-1]) *
                                                                 delta_lambda_dict[transition_name][t_list_index::-1]).sum()

        for transition_name in struct_info1.keys():
            if len(struct_info2[transition_name]) == 1:
                # equation (24)
                other_transition_name = struct_info2[transition_name][0]
                delta_nf_list_dict[transition_name][t_list_index] = \
                    (delta_nd_lists_dict[transition_name][0: t_list_index + 1, 0:t_list_index + 2] *
                     np.exp(-(cumsum_lambda_dict[transition_name][t_list_index::-1].reshape((-1, 1)) +
                              (np.concatenate(([0], cumsum_lambda_dict[other_transition_name][t_list_index::-1])).reshape((1, -1))))) *
                     delta_lambda_dict[transition_name][t_list_index::-1].reshape((-1, 1))).sum()
            if len(struct_info2[transition_name]) == 2:
                # equation (29)
                temp_time = datetime.datetime.now()
                other_transition_name0 = struct_info2[transition_name][0]
                other_transition_name1 = struct_info2[transition_name][1]
                temp1 = delta_nd_lists_dict[transition_name][0]
                delta_nf_list_dict[transition_name][t_list_index] = (temp1 * np.exp(-(cumsum_lambda_dict[transition_name][0:min(
                    L_info[transition_name][1], t_list_index + 1)][::-1].reshape((-1, 1, 1)) + cumsum_lambda_dict[other_transition_name0][
                    0:min(L_info[transition_name][0], t_list_index + 1)][::-1].reshape((1, -1, 1)) + cumsum_lambda_dict[other_transition_name1][
                    0:min(L_info[transition_name][2], t_list_index + 1)][::-1].reshape((1, 1, -1)))) * delta_lambda_dict[transition_name][
                    0:min(L_info[transition_name][1], t_list_index + 1)][::-1].reshape((-1, 1, 1))).sum()
                temp2 = delta_nd_lists_dict[transition_name][1]
                delta_nf_list_dict[transition_name][t_list_index] += (temp2 * np.exp(-(cumsum_lambda_dict[transition_name][0:min(
                    L_info[transition_name][1], t_list_index + 1)][::-1].reshape((-1, 1)) + cumsum_lambda_dict[other_transition_name0][
                    0:min(L_info[transition_name][0], t_list_index + 1)][::-1].reshape((1, -1)))) * delta_lambda_dict[transition_name][
                    0:min(L_info[transition_name][1], t_list_index + 1)][::-1].reshape((-1, 1))).sum()
                temp3 = delta_nd_lists_dict[transition_name][2]
                delta_nf_list_dict[transition_name][t_list_index] += (temp3 * np.exp(-(cumsum_lambda_dict[transition_name][0:min(
                    L_info[transition_name][1] + 0, t_list_index + 1)][::-1].reshape((-1, 1)) + cumsum_lambda_dict[other_transition_name1][
                    0:min(L_info[transition_name][2] + 0, t_list_index + 1)][::-1].reshape((1, -1)))) * delta_lambda_dict[transition_name][
                    0:min(L_info[transition_name][1] + 0, t_list_index + 1)][::-1].reshape((-1, 1))).sum()
                temp4 = delta_nd_lists_dict[transition_name][3]
                delta_nf_list_dict[transition_name][t_list_index] += (temp4 * np.exp(-(cumsum_lambda_dict[transition_name][
                    0:min(L_info[transition_name][1] + 0, t_list_index + 1)][::-1].reshape((-1)))) * delta_lambda_dict[transition_name][
                    0:min(L_info[transition_name][1] + 0, t_list_index + 1)][::-1].reshape((-1))).sum()
                if transition_name == "T6":
                    record_time_list.append((datetime.datetime.now() - temp_time).total_seconds())

        # mean function of the cumulative firing count
        for transition_name in lambda_fun_dict.keys():
            nf_list_dict[transition_name][t_list_index + 1] = nf_list_dict[transition_name][t_list_index] + \
                                                              delta_nf_list_dict[transition_name][t_list_index]
        # mean function of the returned token count
        delta_return_P6 = (delta_nd_lists_dict['T7'][0: t_list_index + 1, 0:t_list_index + 2] *
                             np.exp(-cumsum_lambda_dict['T7'][t_list_index::-1].reshape((-1, 1))) *
                             delta_lambda_dict['T7'][t_list_index::-1].reshape((-1, 1))).sum() - \
                            delta_nf_list_dict['T7'][t_list_index]
        delta_return_P5 = (delta_nd_lists_dict['T7'][0: t_list_index + 1, 0:t_list_index + 2] *
                             np.exp(-(cumsum_lambda_dict['T7'][t_list_index::-1].reshape((-1, 1)) +
                                      np.concatenate(([0], cumsum_lambda_dict['T6'][t_list_index::-1])).reshape((1, -1)))) *
                             (delta_lambda_dict['T7'][t_list_index::-1].reshape((-1, 1)) +
                              np.concatenate(([0], delta_lambda_dict['T6'][t_list_index::-1])).reshape((1, -1)))).sum() + \
                            (delta_nd_lists_dict['T6'][1] *
                             np.exp(-cumsum_lambda_dict['T6'][min(t_list_index, L_info["T6"][1]-1)::-1].reshape((-1, 1))) *
                             delta_lambda_dict['T6'][min(t_list_index, L_info["T6"][1]-1)::-1].reshape((-1, 1))).sum() - \
                            delta_nf_list_dict['T6'][t_list_index] - delta_nf_list_dict['T7'][t_list_index]
        delta_return_P4 = (delta_nd_lists_dict['T5'][0: t_list_index + 1, 0:t_list_index + 2] *
                             np.exp(-(cumsum_lambda_dict['T5'][t_list_index::-1].reshape((-1, 1)) +
                                      np.concatenate(([0], cumsum_lambda_dict['T6'][t_list_index::-1])).reshape((1, -1)))) *
                             (delta_lambda_dict['T5'][t_list_index::-1].reshape((-1, 1)) +
                              np.concatenate(([0], delta_lambda_dict['T6'][t_list_index::-1])).reshape((1, -1)))).sum() * 1 + \
                            (delta_nd_lists_dict['T6'][2] *
                             np.exp(-cumsum_lambda_dict['T6'][min(t_list_index, L_info["T6"][1]-1)::-1].reshape((-1, 1))) *
                             delta_lambda_dict['T6'][min(t_list_index, L_info["T6"][1]-1)::-1].reshape((-1, 1))).sum() * 1 - \
                            delta_nf_list_dict['T5'][t_list_index] * 1 - delta_nf_list_dict['T6'][t_list_index] * 1

        delta_return_P10 = (delta_nd_lists_dict['T11'][0: t_list_index + 1, 0:t_list_index + 2] *
                              np.exp(-(cumsum_lambda_dict['T11'][t_list_index::-1].reshape((-1, 1)) +
                                       np.concatenate(([0], cumsum_lambda_dict['T13'][t_list_index::-1])).reshape((1, -1)))) *
                              (delta_lambda_dict['T11'][t_list_index::-1].reshape((-1, 1)) +
                               np.concatenate(([0], delta_lambda_dict['T13'][t_list_index::-1])).reshape((1, -1)))).sum() + \
                             (delta_nd_lists_dict['T13'][2] *
                              np.exp(-cumsum_lambda_dict['T13'][min(t_list_index, L_info["T13"][1]-1)::-1].reshape((-1, 1))) *
                              delta_lambda_dict['T13'][min(t_list_index, L_info["T13"][1]-1)::-1].reshape((-1, 1))).sum() - \
                             delta_nf_list_dict['T11'][t_list_index] - delta_nf_list_dict['T13'][t_list_index]
        delta_return_P7 = (delta_nd_lists_dict['T15'][0: t_list_index + 1, 0:t_list_index + 2] *
                             np.exp(-(cumsum_lambda_dict['T15'][t_list_index::-1].reshape((-1, 1)) +
                                      np.concatenate(([0], cumsum_lambda_dict['T13'][t_list_index::-1])).reshape((1, -1)))) *
                             (delta_lambda_dict['T15'][t_list_index::-1].reshape((-1, 1)) +
                              np.concatenate(([0], delta_lambda_dict['T13'][t_list_index::-1])).reshape((1, -1)))).sum() * 1 + \
                            (delta_nd_lists_dict['T13'][1] *
                             np.exp(-cumsum_lambda_dict['T13'][min(t_list_index, L_info["T13"][1]-1)::-1].reshape((-1, 1))) *
                             delta_lambda_dict['T13'][min(t_list_index, L_info["T13"][1]-1)::-1].reshape((-1, 1))).sum() * 1 - \
                            delta_nf_list_dict['T15'][t_list_index] * 1 - delta_nf_list_dict['T13'][t_list_index] * 1

        delta_return_P8 = (delta_nd_lists_dict['T16'][0: t_list_index + 1, 0:t_list_index + 2] *
                             np.exp(-(cumsum_lambda_dict['T16'][t_list_index::-1].reshape((-1, 1)) +
                                      np.concatenate(([0], cumsum_lambda_dict['T14'][t_list_index::-1])).reshape((1, -1)))) *
                             (delta_lambda_dict['T16'][t_list_index::-1].reshape((-1, 1)) +
                              np.concatenate(([0], delta_lambda_dict['T14'][t_list_index::-1])).reshape((1, -1)))).sum() + \
                            (delta_nd_lists_dict['T14'][2] *
                             np.exp(-cumsum_lambda_dict['T14'][min(t_list_index, L_info["T14"][1]-1)::-1].reshape((-1, 1))) *
                             delta_lambda_dict['T14'][min(t_list_index, L_info["T14"][1]-1)::-1].reshape((-1, 1))).sum() - \
                            delta_nf_list_dict['T16'][t_list_index] - delta_nf_list_dict['T14'][t_list_index]
        delta_return_P11 = (delta_nd_lists_dict['T12'][0: t_list_index + 1, 0:t_list_index + 2] *
                              np.exp(-(cumsum_lambda_dict['T12'][t_list_index::-1].reshape((-1, 1)) +
                                       np.concatenate(([0], cumsum_lambda_dict['T14'][t_list_index::-1])).reshape((1, -1)))) *
                              (delta_lambda_dict['T12'][t_list_index::-1].reshape((-1, 1)) +
                               np.concatenate(([0], delta_lambda_dict['T14'][t_list_index::-1])).reshape((1, -1)))).sum() * 1 + \
                             (delta_nd_lists_dict['T14'][1] *
                              np.exp(-cumsum_lambda_dict['T14'][min(t_list_index, L_info["T14"][1]-1)::-1].reshape((-1, 1))) *
                              delta_lambda_dict['T14'][min(t_list_index, L_info["T14"][1]-1)::-1].reshape((-1, 1))).sum() * 1 - \
                             delta_nf_list_dict['T12'][t_list_index] * 1 - delta_nf_list_dict['T14'][t_list_index] * 1

        # mean function of the cumulative returned token count
        return_nd['P4'][t_list_index] = delta_return_P4 + return_nd['P4'][t_list_index - 1]
        return_nd['P5'][t_list_index] = delta_return_P5 + return_nd['P5'][t_list_index - 1]
        return_nd['P6'][t_list_index] = delta_return_P6 + return_nd['P6'][t_list_index - 1]
        return_nd['P10'][t_list_index] = delta_return_P10 + return_nd['P10'][t_list_index - 1]
        return_nd['P7'][t_list_index] = delta_return_P7 + return_nd['P7'][t_list_index - 1]
        return_nd['P8'][t_list_index] = delta_return_P8 + return_nd['P8'][t_list_index - 1]
        return_nd['P11'][t_list_index] = delta_return_P11 + return_nd['P11'][t_list_index - 1]

        # mean function of the cumulative enabling count
        nd_list_dict["T0"][t_list_index + 2] = mP0 + nf_list_dict['T5'][t_list_index + 1] + nf_list_dict['T6'][t_list_index + 1] + \
                                               nf_list_dict['T7'][t_list_index + 1] + nf_list_dict['T11'][t_list_index + 1] + \
                                               nf_list_dict['T13'][t_list_index + 1] + nf_list_dict['T12'][t_list_index + 1] + \
                                               nf_list_dict['T14'][t_list_index + 1]
        nd_list_dict['T1'][t_list_index + 2] = nf_list_dict['T0'][t_list_index + 1] * pP1T1
        nd_list_dict['T2'][t_list_index + 2] = nf_list_dict['T0'][t_list_index + 1] * pP1T2
        nd_list_dict['T8'][t_list_index + 2] = nf_list_dict['T0'][t_list_index + 1] * pP1T8
        nd_list_dict['T4'][t_list_index + 2] = nf_list_dict['T1'][t_list_index + 1] * pP2T4
        nd_list_dict['T10'][t_list_index + 2] = nf_list_dict['T1'][t_list_index + 1] * pP2T10

        nd_list_dict['T3'][t_list_index + 2] = mP3 + nf_list_dict['T3'][t_list_index + 1]
        nd_list_dict['T9'][t_list_index + 2] = mP9 + nf_list_dict['T9'][t_list_index + 1]

        nd_list_dict['T5'][t_list_index + 2] = nf_list_dict['T2'][t_list_index + 1] + return_nd['P4'][t_list_index]
        nd_list_dict['T6'][t_list_index + 2] = min(nf_list_dict['T2'][t_list_index + 1] + return_nd['P4'][t_list_index],
                                                   (nf_list_dict['T3'][t_list_index + 1] + return_nd['P5'][t_list_index]))
        nd_list_dict['T7'][t_list_index + 2] = min((nf_list_dict['T3'][t_list_index + 1] + return_nd['P5'][t_list_index]),
                                                   (nf_list_dict['T4'][t_list_index + 1] + return_nd['P6'][t_list_index]))
        nd_list_dict['T11'][t_list_index + 2] = nf_list_dict['T8'][t_list_index + 1] + return_nd['P10'][t_list_index]
        nd_list_dict['T13'][t_list_index + 2] = min(nf_list_dict['T8'][t_list_index + 1] + return_nd['P10'][t_list_index],
                                                    nf_list_dict['T9'][t_list_index + 1] + return_nd['P7'][t_list_index])
        nd_list_dict['T15'][t_list_index + 2] = nf_list_dict['T9'][t_list_index + 1] + return_nd['P7'][t_list_index]
        nd_list_dict['T16'][t_list_index + 2] = nf_list_dict['T9'][t_list_index + 1] * 0.1 + return_nd['P8'][t_list_index]
        nd_list_dict['T14'][t_list_index + 2] = min(nf_list_dict['T9'][t_list_index + 1] * 0.1 + return_nd['P8'][t_list_index],
                                                    nf_list_dict['T10'][t_list_index + 1] + return_nd['P11'][t_list_index])
        nd_list_dict['T12'][t_list_index + 2] = nf_list_dict['T10'][t_list_index + 1] + return_nd['P11'][t_list_index]
        # For transitions where the firing rate is always 0, their enabling counts are set to 0.
        for transition_name in delta_lambda_dict.keys():
            if delta_lambda_dict[transition_name].sum() == 0:
                nd_list_dict[transition_name][t_list_index + 2] = 0

        # the cumulative provided token count
        place_num_list_dict['P4'][t_list_index + 2] = nf_list_dict['T2'][t_list_index + 1] + return_nd['P4'][t_list_index]
        place_num_list_dict['P5'][t_list_index + 2] = nf_list_dict['T3'][t_list_index + 1] + return_nd['P5'][t_list_index]
        place_num_list_dict['P6'][t_list_index + 2] = nf_list_dict['T4'][t_list_index + 1] + return_nd['P6'][t_list_index]
        place_num_list_dict['P10'][t_list_index + 2] = nf_list_dict['T8'][t_list_index + 1] + return_nd['P10'][t_list_index]
        place_num_list_dict['P7'][t_list_index + 2] = nf_list_dict['T9'][t_list_index + 1] + return_nd['P7'][t_list_index]
        place_num_list_dict['P8'][t_list_index + 2] = nf_list_dict['T9'][t_list_index + 1] * 0.1 + return_nd['P8'][t_list_index]
        place_num_list_dict['P11'][t_list_index + 2] = nf_list_dict['T10'][t_list_index + 1] + return_nd['P11'][t_list_index]
    last_time = datetime.datetime.now()
    print('Methods 1 and 2 completed!')
    print('total process time: {:.5f}s'.format((last_time-first_time).total_seconds()))
    print()
    return nf_list_dict


if __name__ == "__main__":
    nf_list_dict = example_3()
    print("cumulative firing counts of transitions:")
    for transition_name in nf_list_dict.keys():
        print(nf_list_dict[transition_name].tolist()[-1], transition_name, nf_list_dict[transition_name].tolist())
