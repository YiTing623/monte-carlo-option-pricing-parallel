#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;

// -----------------------------------------------------------------
// Function: simulateOptionPayoff
// Purpose:  Simulates a single price path for the underlying asset
//           and calculates the payoff of a European call option.
// -----------------------------------------------------------------
double simulateOptionPayoff(double S0, double K, double r, double sigma, double T, int steps, std::default_random_engine &generator) {
    double dt = T / steps;
    double S = S0;
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < steps; i++) {
        double Z = distribution(generator);
        S *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
    }
    return std::max(S - K, 0.0);
}

// -----------------------------------------------------------------
// Function: monteCarloPrice (Sequential)
// Purpose:  Uses Monte Carlo simulation to calculate the price of a
//           European call option by averaging discounted payoffs.
// -----------------------------------------------------------------
double monteCarloPrice(int numSimulations, double S0, double K, double r, double sigma, double T, int steps) {
    double totalPayoff = 0.0;
    std::random_device rd;
    std::default_random_engine generator(rd());

    for (int i = 0; i < numSimulations; i++) {
        double payoff = simulateOptionPayoff(S0, K, r, sigma, T, steps, generator);
        totalPayoff += payoff;
    }
    double discountedPayoffAvg = (totalPayoff / numSimulations) * std::exp(-r * T);
    return discountedPayoffAvg;
}

// -----------------------------------------------------------------
// Function: blackScholesPrice
// Purpose:  Computes the Black–Scholes price for a European call option.
// -----------------------------------------------------------------
double blackScholesPrice(double S0, double K, double r, double sigma, double T) {
    double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);

    auto phi = [](double x) {
        return 0.5 * std::erfc(-x / std::sqrt(2));
    };

    double callPrice = S0 * phi(d1) - K * std::exp(-r * T) * phi(d2);
    return callPrice;
}

// -----------------------------------------------------------------
// Main function
// -----------------------------------------------------------------
int main() {
    // Option and Simulation Parameters
    double S0 = 100.0;      // Initial stock price
    double K = 100.0;       // Strike price
    double r = 0.05;        // Risk-free interest rate (5%)
    double sigma = 0.2;     // Volatility (20%)
    double T = 1.0;         // Time to maturity: 1 year
    int steps = 252;        // Number of time steps (daily steps for 1 year)

    // Define simulation counts to test
    int simulationCounts[6] = {10000, 50000, 100000, 500000, 1000000, 5000000};

    // Calculate the Black–Scholes price (constant for these parameters)
    double bsPrice = blackScholesPrice(S0, K, r, sigma, T);

    // Print the constant Black-Scholes Price once
    std::cout << "Constant Black-Scholes Price for European Call Option: " << bsPrice << "\n\n";

    // Iterate over the simulation counts
    for (int i = 0; i < 6; i++) {
        int numSimulations = simulationCounts[i];
        std::cout << "---------------------------------------------\n";
        std::cout << "numSimulations: " << numSimulations << "\n";

        auto start = std::chrono::high_resolution_clock::now();
        double mcPrice = monteCarloPrice(numSimulations, S0, K, r, sigma, T, steps);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        std::cout << "Runtime: " << elapsed.count() << " seconds\n";
        std::cout << "Monte Carlo Price: " << mcPrice << "\n";
        std::cout << "Black-Scholes Price: " << bsPrice << "\n";

        double relativeError = std::abs(mcPrice - bsPrice) / bsPrice;
        std::cout << "Relative Error: " << (relativeError * 100) << " %\n\n";
    }
    return 0;
}
