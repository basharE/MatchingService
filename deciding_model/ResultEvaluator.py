import logging
from collections import Counter
import numpy as np


def get_result_from_trainer(trainer, input_data):
    final_list_for_prediction = list()
    for value in input_data.values():
        final_list_for_prediction = final_list_for_prediction + value[0:, 1].tolist()

    predict_result_per_classifier = list()
    for i in range(len(trainer.classifiers)):
        predict_result_per_classifier = predict_result_per_classifier + (
            trainer.classifiers[i].predict([final_list_for_prediction])).tolist()

    element_counts = Counter(predict_result_per_classifier)

    most_common_element = element_counts.most_common(1)[0][0]

    print("Element with the maximum occurrence:", most_common_element)
    return most_common_element


class ResultEvaluator:
    def evaluate(self, input_data, _class):
        candidates = self.get_candidates(input_data)
        _sum = self.get_sum(input_data)
        _avg = _sum / len(input_data)
        logging.info(f"The Average Result: {_avg}")
        # Find the maximum value and its index
        max_value = np.max(_avg[0:, 1])
        max_index = np.argmax(_avg[0:, 1])
        # if _class is not None:
        #     collection = connect_to_collection(get_database_uri_from_conf(), get_database_name_from_conf(),
        #                                        get_classifiers_collection_name_from_conf())
        #
        #     v_ = input_data.values()
        #     # Initialize row_ as an empty NumPy array
        #     row_ = np.array([])
        #
        #     # Loop through the elements in v_ and concatenate them to row_
        #     for i in v_:
        #         row_ = np.concatenate((row_, i[0:, 1]))
        #
        #     # Value to add
        #     value_to_add = _class
        #
        #     # Convert the value to add into a NumPy array with the same shape
        #     value_array = np.array([value_to_add])
        #
        #     # Concatenate the value array to the original array using numpy.concatenate()
        #     my_array = np.concatenate((row_, value_array))
        #     array_dict = {'data': my_array.tolist()}
        #
        #     collection.insert_one(array_dict)

        # trainer = ResultTrainer()
        # return get_result_from_trainer(trainer, input_data)

        if max_value > 0.87:
            return max_index
        else:
            return None
        # response = self.decide_on_prediction_result(_sum, candidates, input_data)
        # multinomial_nb_result = input_data["MultinomialNB"]
        # if response is not None:
        #     if isinstance(response, list):
        #         if multinomial_nb_result[0:, 1][response[0]] > 0.7:
        #             return response[0]
        #     else:
        #         if multinomial_nb_result[0:, 1][response] > 0.7:
        #             return response
        # return None

    def decide_on_prediction_result(self, _sum, candidates, results_dict):
        max_value_sum = np.max(_sum[0:, 1])
        max_indices_sum = [i for i, value in enumerate(_sum[0:, 1]) if value == max_value_sum]
        if len(max_indices_sum) == 1:
            candidate_max = max_indices_sum
        mlp_classifier_result = results_dict['MLPClassifier']
        # Get the index of the maximum value using numpy.argmax
        max_index = np.argmax(mlp_classifier_result[0:, 1])
        # Get the maximum value
        maximum = mlp_classifier_result[0:, 1][max_index]
        if (maximum == 1 and np.isin(max_index, candidates)) and (candidate_max == max_index) and (
                np.isin(candidate_max, candidates)):
            return max_index

        if np.isin(candidate_max, candidates):
            return candidate_max

        random_forest_classifier_result = results_dict['RandomForestClassifier']

        indices_above_080 = np.argwhere(random_forest_classifier_result[0:, 1] >= 0.80)

        intersection = np.intersect1d(indices_above_080, candidates)

        if intersection.size > 1:
            if intersection.size == 2:
                if mlp_classifier_result[0:, 1][candidates[0]] > mlp_classifier_result[0:, 1][candidates[1]] and \
                        random_forest_classifier_result[0:, 1][candidates[0]] > random_forest_classifier_result[0:, 1][
                    candidates[1]]:
                    return candidates[0]
                if mlp_classifier_result[0:, 1][candidates[1]] > mlp_classifier_result[0:, 1][candidates[0]] and \
                        random_forest_classifier_result[0:, 1][candidates[1]] > random_forest_classifier_result[0:, 1][
                    candidates[0]]:
                    return candidates[1]
            return -1
        if intersection.size == 1:
            return intersection

        # Get the index of the maximum value using numpy.argmax
        max_index = np.argmax(random_forest_classifier_result[0:, 1])
        # Get the maximum value
        maximum = random_forest_classifier_result[0:, 1][max_index]
        if maximum > 0.75 and np.isin(candidates, max_index):
            return max_index
        return None

    def get_candidates(self, results_dict):
        candidates_list = list()
        final_candidates_list = list()
        decision_tree_classifier_result = results_dict['DecisionTreeClassifier']
        max_value = np.max(decision_tree_classifier_result[0:, 1])
        max_indices = [i for i, value in enumerate(decision_tree_classifier_result[0:, 1]) if value == max_value]
        for i in max_indices:
            if decision_tree_classifier_result[0:, 1][i] == 1:
                candidates_list.append(i)

        k_neighbors_classifier_result = results_dict['KNeighborsClassifier']
        for i in candidates_list:
            if k_neighbors_classifier_result[0:, 1][i] == 1:
                final_candidates_list.append(i)

        return np.array(final_candidates_list)

    def get_sum(self, input_data):
        values_view = input_data.values()

        # Convert the view to a list
        values_list = list(values_view)

        # Convert the list of arrays to a NumPy array
        values_array = np.array(values_list)

        # Sum the arrays element-wise along the first axis (axis=0)
        sum_result = np.sum(values_array, axis=0)

        return sum_result
