/* Filename: VectorStuff.cpp
 * Author: Colton Townsend / cxt180021
 * Date: Created 2/3/2023
 * Procedures:
 * main - 
 */
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <math.h>
#include <float.h>

// Forward declaration
void printVector(std::vector<double> &target);
double sumVector(std::vector<double> &target);
double meanVector(std::vector<double> &target);
double medianVector(std::vector<double> &target);
void rangeVector(std::vector<double> &target);
void print_stats(std::vector<double> &target);
double covar(std::vector<double> &rm, std::vector<double> &medv);
double cor(std::vector<double> &rm, std::vector<double> &medv);

int main(int argc, char** argv){
std::ifstream inFS;
std::string line;
std::string rm_in, medv_in;
const int MAX_LEN = 1000;
std::vector<double> rm(MAX_LEN);
std::vector<double> medv(MAX_LEN);
std::cout << "Opening file Boston.csv." << std::endl;
inFS.open("Boston.csv");
if (!inFS.is_open()){
	std::cout << "Could not open file Boston.csv." << std::endl;
	return 1;
}
std::cout << "Reading line 1" << std::endl;
getline(inFS, line);

std::cout << "heading: " << line << std::endl;

int numObservations = 0;
while (inFS.good()){
	getline(inFS, rm_in, ',');
	getline(inFS, medv_in, '\n');
	
	rm.at(numObservations) = stof(rm_in);
	medv.at(numObservations) = stof(medv_in);
	
	numObservations++;
}

rm.resize(numObservations);
medv.resize(numObservations);

std::cout << "new length " << rm.size() << std::endl;

std::cout << "Closing file Boston.csv."  << std::endl;
inFS.close();

std::cout << "Number of records: " << numObservations << std::endl;

std::cout << "\nStats for rm" << std::endl;
print_stats(rm);

std::cout << "\nStats for medv" << std::endl;
print_stats(medv);

std::cout << "\n Covariance = " << covar(rm,medv) << std::endl;

std::cout << "\n Correlation = " << cor(rm,medv) << std::endl;

std::cout << "\nProgram terminated.";

return 0;
}

void print_stats(std::vector<double> &target){
	std::cout << "Sum of vector: ";
	std::cout << sumVector(target) << std::endl;
	std::cout << "Mean of vector: ";
	std::cout << meanVector(target) << std::endl;
	std::cout << "Median of vector: ";
	std::cout << medianVector(target) << std::endl;
	std::cout << "Range of vector: ";
	rangeVector(target);
}

void printVector(std::vector<double> &target){
   std::cout << "Printing vector: ";
   for(int i=0; i < target.size(); i++)
   std::cout << target.at(i) << ' ';
}

double sumVector(std::vector<double> &target){
	double sum = 0;
	for(int i=0; i < target.size(); i++)
    sum += target.at(i);
	return sum;
}

double meanVector(std::vector<double> &target){
	return (sumVector(target)/target.size());
}

double medianVector(std::vector<double> &target){
	if (target.size() == 0){
		return 0;
	} else {
		sort(target.begin(), target.end());
		if (target.size() % 2 == 0){
			return (target[target.size() / 2 - 1] + target[target.size() / 2]) / 2;
		} else {
			return target[target.size() /  2];
		}
	}
}

void rangeVector(std::vector<double> &target){
	double max = DBL_MIN;
	double min = DBL_MAX;
	for(int i=0; i < target.size(); i++){
		if (target[i] > max){
			max = target[i];
		}
		if (target[i] < min){
			min = target[i];
		}
	}
	std::cout << "\n    Min: " << min;
	std::cout << "\n    Max: " << max << std::endl;
}

double covar(std::vector<double> &rm, std::vector<double> &medv){
	if (rm.size() != medv.size()){
		std::cout << "Covariance matrix relies on the idea that each observation in rm corresponds to an observation in medv.\n Since the vectors are not of equal size, we cannot do a covariance matrix.\n";
		return 0;
	}
	double rmMean = meanVector(rm);
	double medvMean = meanVector(medv);
	int sum = 0;
	
	for(int i=0; i < rm.size(); i++) // Doesn't matter if we use rm.size or medv.size because the vectors are of equal size
		sum += ((rm[i]-rmMean)*(medv[i]-medvMean));
	return (sum/(rm.size()-1));
}

double cor(std::vector<double> &rm, std::vector<double> &medv){
	double sum = 0;
	double rmSD, medvSD;
	double rmMean = meanVector(rm);
	double medvMean = meanVector(medv);
	
	for(int i=0; i < rm.size(); i++)
		sum += ((rm[i]-rmMean)*(rm[i]-rmMean));
	sum = sum/(rm.size()-1);
	
	rmSD = sqrt(sum);
	
	sum = 0;
	for(int i=0; i < medv.size(); i++)
		sum += ((rm[i]-medvMean)*(rm[i]-medvMean));
	sum = sum/(medv.size()-1);
	
	medvSD = sqrt(sum);
	
	return (covar(rm, medv)/(rmSD*medvSD));
}
