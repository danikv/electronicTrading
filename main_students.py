import hw1_part1 as st
import Timer


def main():

    total_timer = Timer.Timer('Total running time')

    ###########################################################################
    #######################            Part A           #######################
    ###########################################################################
    time = 1345521600

    """Read the data files with the class ModelData into the a data object"""
    part_a_timer = Timer.Timer('Part A')
    data_reading = Timer.Timer('Reading Data')
    data = st.ModelData('data_students.txt', time)
    data_reading.stop()

    histogram_timer = Timer.Timer('Part A1')
    st.calculate_distribution('data_students.txt')
    histogram_timer.stop()

    features_timer = Timer.Timer('Part A2')
    features_dict = st.G_features(data, time)
    features_timer.stop()

    part_a_timer.stop()

    ###########################################################################
    #######################            Part B           #######################
    ###########################################################################

    part_b_timer = Timer.Timer('Part B')
    k = 6
    time = 1307221600

    """Read the data files with the class ModelData into the a data object"""
    data_reading = Timer.Timer('Reading Data')
    data = st.ModelData('data_students.txt', time)
    data_reading.stop()

    ####################### unweighted undirected graph #######################

    model0_timer = Timer.Timer('unweighted undirected graph')

    unweighted_H_t = st.create_unweighted_H_t(data.train, time)
    test_predictions_list = st.run_k_iterations(unweighted_H_t, k, mode='undirected unweighted')
    eval_timer = Timer.Timer('Error calculation')

    eval_res = st.calc_error(test_predictions_list, data.test_x,  mode='undirected unweighted')
    print(f'unweighted undirected graph Precision {eval_res[0]}, Recall {eval_res[1]}\n')
    eval_timer.stop()

    model0_timer.stop()

    ####################### weighted undirected graph #######################
    model1_timer = Timer.Timer('weighted undirected graph')

    weighted_H_t = st.create_weighted_H_t(data.train, time)

    test_predictions_list = st.run_k_iterations(weighted_H_t, k, mode='undirected weighted')

    eval_timer = Timer.Timer('Error Calculation')
    eval_res = st.calc_error(test_predictions_list, data.test_x, mode='undirected weighted')
    print(f'weighted undirected graph Precision {eval_res[0]}, Recall {eval_res[1]}\n')
    eval_timer.stop()

    model1_timer.stop()

    ####################### unweighted directed graph #######################
    model2_timer = Timer.Timer('unweighted directed graph')

    G_t = st.create_unweighted_G_t(data.train, time)
    test_predictions_list = st.run_k_iterations(G_t, k, mode='directed')

    eval_timer = Timer.Timer('Error Calculation')
    eval_res = st.calc_error(test_predictions_list, data.test_x,  mode='directed')
    print(f'uneighted directed graph Precision {eval_res[0]}, Recall {eval_res[1]}\n')
    eval_timer.stop()

    model2_timer.stop()
    part_b_timer.stop()

    total_timer.stop()


if __name__ == '__main__':
    main()
