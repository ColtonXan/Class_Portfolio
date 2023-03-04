/* Filename: VectorStuff.cpp
 * Author: Colton Townsend / cxt180021
 * Date: Created 3/2/2023
 * Procedures:
 * read_csv - Takes in a csv file as input (filename) and will read rows up to a limit determined by
 * observationlimit. First column is skipped, so data[0] correlates to the pclass column.
 * sum - Returns sum of all cells in a vector.
 * sum - Returns product of all cells in a vector.
 * sigmoid - Returns sigmoid of the double.
 * accuracy - Returns the percent of rows that are equal between two vectors.
 * main - 
 */
 
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <unordered_map>

std::vector<std::vector<double>> read_csv(std::string filename, int observationlimit){
	std::ifstream inFS(filename); // Open the CSV
	std::vector<std::vector<double>> data; // Data Matrix
    std::string line;
	int count = 0; // We can only use the first observationlimit observations
	int row_num = 0;
	while (getline(inFS, line) && count < observationlimit) { // Loop over rows & fill matrix until limit is reached
		if (row_num != 0){ // Skipping the header row
			std::vector<double> row;
			std::stringstream ss(line);
			std::string cell;
			//getline(ss, cell, ','); // legacy way to skip the first column
			int col_num = 0;
			while (getline(ss, cell, ',')) {
				if (col_num != 0){
					try{
						row.push_back(stod(cell));
					} catch (const std::invalid_argument& e) {
						std::cout << "Invalid argument caught: " << e.what() << std::endl; // fires off if a non-numeric cell interrupts the function
						row.clear(); // in such case, clear the row and try to continue
						break;
					}
				}
				col_num++;
			}
			if (!row.empty()){
			data.push_back(row);
			count++;
			}
		} 
		row_num++;
    }
    return data; // Return the matrix
}

double sum(std::vector<double> v){
    double s = 0.0;
    for (double x : v) {
        s += x;
    }
    return s;
}

double dot(std::vector<double> v1, std::vector<double> v2){
    double d = 0.0;
    for (int i = 0; i < v1.size(); i++) {
        d += v1[i] * v2[i];
    }
    return d;
}

double sigmoid(double x){
	return 1/(1+exp(-x));
}

double accuracy(std::vector<double> y_true, std::vector<double> y_pred) {
    int n = y_true.size();
    int num_correct = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] == round(y_pred[i])) { // Compare
            num_correct++;
        }
    }
    return static_cast<double>(num_correct) / n;
}

double sensitivity(std::vector<double> y_true, std::vector<double> y_pred) {
    int n = y_true.size();
    int true_positives = 0;
    int false_negatives = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] == 1 && round(y_pred[i]) == 1) { // Compare (case of true)
            true_positives++;
        } else if (y_true[i] == 1 && round(y_pred[i]) == 0) { // Compare (case of false)
            false_negatives++;
        }
    }
    if (true_positives + false_negatives == 0) { // Case of any empty vector
        return 0;
    } else {
        return static_cast<double>(true_positives) / (true_positives + false_negatives);
    }
}

double specificity(std::vector<double> y_true, std::vector<double> y_pred) {
    int n = y_true.size();
    int true_negatives = 0;
    int false_positives = 0;
    for (int i = 0; i < n; i++) {
        if (y_true[i] == 0 && round(y_pred[i]) == 0) { // Compare (case of true)
            true_negatives++;
        } else if (y_true[i] == 0 && round(y_pred[i]) == 1) { // Compare (case of false)
            false_positives++;
        }
    }
    if (true_negatives + false_positives == 0) { // Case of any empty vector
        return 0;
    } else {
        return static_cast<double>(true_negatives) / (true_negatives + false_positives);
    }
}

void naive_bayes(const std::vector<std::vector<double>>& features, const std::vector<double>& target) {
    // Calculate prior probabilities of each class by dividing the number of instances of each class by the total number of instances
    std::unordered_map<double, double> prior_counts;
    for (double t : target) {
        if (prior_counts.find(t) == prior_counts.end()) {
            prior_counts[t] = 0.0;
        }
        prior_counts[t] += 1.0;
    }
    std::unordered_map<double, double> prior_probs;
    double total_rows = target.size();
    for (const auto& [t, count] : prior_counts) {
        prior_probs[t] = count / total_rows;
    }

    // For each instance in the set, calculate conditional probabilities by Bayes formula
    std::unordered_map<double, std::vector<double>> mean_values;
    std::unordered_map<double, std::vector<double>> std_dev_values;
    for (int i = 0; i < features[0].size(); ++i) {
        double feature_mean = 0.0;
        double feature_std_dev = 0.0;
        for (int j = 0; j < features.size(); ++j) {
            double f = features[j][i];
            feature_mean += f;
        }
        feature_mean /= total_rows;
        for (int j = 0; j < features.size(); ++j) {
            double f = features[j][i];
            feature_std_dev += pow(f - feature_mean, 2);
        }
        feature_std_dev = sqrt(feature_std_dev / (total_rows - 1));

        for (int j = 0; j < features.size(); ++j) {
            double t = target[j];
            double f = features[j][i];
            if (mean_values.find(t) == mean_values.end()) {
                mean_values[t] = std::vector<double>(features[0].size(), 0.0);
            }
            mean_values[t][i] += f;
            if (std_dev_values.find(t) == std_dev_values.end()) {
                std_dev_values[t] = std::vector<double>(features[0].size(), 0.0);
            }
            std_dev_values[t][i] += pow(f - feature_mean, 2);
        }
    }
    for (const auto& [t, count] : prior_counts) {
        for (int i = 0; i < features[0].size(); ++i) {
            mean_values[t][i] /= count;
            std_dev_values[t][i] = sqrt(std_dev_values[t][i] / (count - 1));
        }
    }

    // Predict the class with the highest conditional probability for each instance
	std::vector<double> y_pred;
	int correct_guesses = 0;
    for (int i = 0; i < features.size(); i++) {
		double real_target = target[i];
        double max_prob = -INFINITY; // Class probability should be updated per loop
        double predicted_target = -1.0; // Initialize to a value outside of the range of possible predictions
        for (const auto& [t, prob] : prior_probs) {
            double class_prob = log(prob);
            for (int j = 0; j < features[0].size(); j++) {
                double f = features[i][j];
                double mean = mean_values[t][j];
                double std_dev = std_dev_values[t][j];
                double cond_prob = -log(std_dev) - pow(f - mean, 2) / (2 * pow(std_dev, 2));
                class_prob += cond_prob;
            }
            if (class_prob > max_prob) {
                max_prob = class_prob;
                predicted_target = t; // Change the predicted target
            }
        }
		// Print the prediction every 100 iterations
        if (i % 100 == 0) {
			std::cout << "Row " << i << ": predicted target = " << predicted_target << " Real target = " << real_target << std::endl;
		}
		if (predicted_target == real_target){
			correct_guesses++;
		}
		y_pred.push_back(predicted_target); // Push the predicted value to our vector of predicted values
	}
	// Evaluate the accuracy with our predicted values vector and our real values vector
	std::cout << "Correct Guesses: " << correct_guesses << " out of " << target.size() << std::endl;
	std::cout << "Naïve Bayes Accuracy: " << accuracy(target, y_pred) << std::endl;
	std::cout << "Naïve Bayes Sensitivity: " << sensitivity(target, y_pred) << std::endl;
	std::cout << "Naïve Bayes Specificity: " << specificity(target, y_pred) << std::endl;
}

int main(int argc, char** argv){
	// Start clock
	auto start = std::chrono::high_resolution_clock::now();
	
	// Read the CSV file
    std::vector<std::vector<double>> data = read_csv("titanic_project.csv", 800); // train/pred data
	
	std::vector<double> x, y;
	std::vector<std::vector<double>> features; // matrix of only dependents
    for (std::vector<double> row : data) {
        x.push_back(row[2]); // sex (independent)
        y.push_back(row[1]); // survived (dependent)
		
		std::vector<double> feature_row = {row[0], row[2], row[3]}; // columns of dependents
		features.push_back(feature_row); // (for naive bayes) this is easier than removing "sex" from the "data" matrix
    }
	
	 // Initialize the parameters of the logistic regression model
    double w0 = 0.0; // intercept
    double w1 = 0.0; // coefficient
    double learning_rate = 0.01;
    int num_iterations = 10000;

    // Perform gradient descent to learn the parameters
    for (int i = 0; i < num_iterations; i++) {
        double y_pred;
        double cost = 0.0;
        double dw0 = 0.0;
        double dw1 = 0.0;
        for (int j = 0; j < x.size(); j++) {
            // Compute the predicted value of y
            y_pred = sigmoid(w0 + w1 * x[j]);

            // Compute the cost function and the gradients
            cost += -y[j] * log(y_pred) - (1 - y[j]) * log(1 - y_pred);
            dw0 += (y_pred - y[j]);
            dw1 += (y_pred - y[j]) * x[j];
        }
        cost /= x.size();
        dw0 /= x.size();
        dw1 /= x.size();

        // Update the parameters using gradient descent
        w0 = w0 - learning_rate * dw0;
        w1 = w1 - learning_rate * dw1;

        // Print the cost function every 1000 iterations
        if (i % 1000 == 0) {
            std::cout << "Iteration " << i << ", cost = " << cost << std::endl;
        }
    }

    // Print the final parameters of the model
    std::cout << "Final parameters: Intercept = " << w0 << ", Coefficient = " << w1 << std::endl;

    // Predict the probability of survival for a male and a female
    double male = sigmoid(w0 + w1 * 1);
    double female = sigmoid(w0 + w1 * 0);
    std::cout << "Probability of survival for a male: " << male << std::endl;
    std::cout << "Probability of survival for a female: " << female << std::endl;

	std::vector<double> y_pred;
	for (int j = 0; j < x.size(); j++) {
		y_pred.push_back(sigmoid(w0 + w1 * x[j]));
	}
	
	// Evaluate accuracy, sensitivity, and specificity
	std::cout << "Accuracy: " << accuracy(y, y_pred) << std::endl;
	std::cout << "Sensitivity: " << sensitivity(y, y_pred) << std::endl;
	std::cout << "Specificity: " << specificity(y, y_pred) << std::endl;
	
	// (B) Naive Bayes
	naive_bayes(features, y); // All of that work in main for logistic regression... and here we have one line for naive bayes! :)
	
	// Stop clock
	auto end = std::chrono::high_resolution_clock::now();
	auto duration =  std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "Execution time: " << duration << " ms" << std::endl;
	return 0;
	
}