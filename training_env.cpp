// training_env.cpp
//
// This file implements a very simple tabular Q-learning algorithm for the
// headless Mars lander environment exposed via rl_apis in lander.h and
// lander.cpp.  It discretises the continuous altitude and vertical
// velocity into coarse bins and learns a throttle policy over five
// discrete throttle settings.  This example is deliberately simple
// and primarily serves to illustrate how to drive the environment via
// the RL helper functions.  For more sophisticated control (e.g. PPO
// or SAC with neural networks) see the comments in lander.cpp.

#include "lander.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Discretisation parameters
static const int ALT_BINS = 12;       // number of altitude bins (0–12 km)
static const int VZ_BINS  = 15;       // number of vertical speed bins (-100–+100 m/s)
static const int NUM_ACTIONS = 5;     // number of discrete throttle levels
static const double ALT_MAX = 12000.0; // maximum altitude considered (m)
static const double VZ_MAX  = 100.0;   // maximum |vz| considered (m/s)

// Q-value table indexed by [alt_bin][vz_bin][action]
static double Q[ALT_BINS][VZ_BINS][NUM_ACTIONS];

// Discretise continuous altitude (m) and vertical speed (m/s) into bins
static inline void discretise(double alt, double vz, int &alt_bin, int &vz_bin)
{
    // Clamp values to the expected ranges
    if (alt < 0.0) alt = 0.0;
    if (alt > ALT_MAX) alt = ALT_MAX;
    if (vz < -VZ_MAX) vz = -VZ_MAX;
    if (vz > VZ_MAX) vz = VZ_MAX;
    // Compute bin indices
    double alt_bin_size = ALT_MAX / static_cast<double>(ALT_BINS);
    double vz_bin_size  = (2.0 * VZ_MAX) / static_cast<double>(VZ_BINS);
    alt_bin = static_cast<int>(alt / alt_bin_size);
    vz_bin  = static_cast<int>((vz + VZ_MAX) / vz_bin_size);
    if (alt_bin >= ALT_BINS) alt_bin = ALT_BINS - 1;
    if (vz_bin >= VZ_BINS) vz_bin = VZ_BINS - 1;
}

// ε-greedy action selection from a given state
static inline int choose_action(int alt_bin, int vz_bin, double epsilon)
{
    // With probability epsilon choose a random action
    if (((double)std::rand() / (double)RAND_MAX) < epsilon) {
        return std::rand() % NUM_ACTIONS;
    }
    // Otherwise choose the action with highest Q-value
    int best = 0;
    double best_val = Q[alt_bin][vz_bin][0];
    for (int a = 1; a < NUM_ACTIONS; ++a) {
        if (Q[alt_bin][vz_bin][a] > best_val) {
            best_val = Q[alt_bin][vz_bin][a];
            best = a;
        }
    }
    return best;
}

int main()
{
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    // Hyperparameters
    const double alpha  = 0.1;   // learning rate
    const double gamma  = 0.99;  // discount factor
    double epsilon      = 0.2;   // exploration rate
    const int max_episodes = 200; // number of training episodes
    const int max_steps    = 10000; // maximum steps per episode
    const double dt        = 0.05;  // simulation time step (s)

    // Initialise Q-table to zero
    for (int i = 0; i < ALT_BINS; ++i)
        for (int j = 0; j < VZ_BINS; ++j)
            for (int k = 0; k < NUM_ACTIONS; ++k)
                Q[i][j][k] = 0.0;

    // Training loop
    for (int episode = 0; episode < max_episodes; ++episode) {
        // Reset environment to a new random start state
        rl_reset(std::rand());
        // Get initial observation
        std::vector<float> obs = rl_observation();
        // Convert normalised observations back to approximate SI units
        double alt = static_cast<double>(obs[0]) * 10000.0; // altitude in m (see lander.cpp scaling)
        double vz  = static_cast<double>(obs[1]) * 100.0;    // vertical speed in m/s
        int alt_bin, vz_bin;
        discretise(alt, vz, alt_bin, vz_bin);
        double total_reward = 0.0;
        bool done = false;
        for (int step = 0; step < max_steps; ++step) {
            // Choose throttle action via ε-greedy policy
            int action = choose_action(alt_bin, vz_bin, epsilon);
            // Map discrete action index to throttle value [0,1]
            double throttle_cmd = static_cast<double>(action) / static_cast<double>(NUM_ACTIONS - 1);
            // Apply action and step environment
            rl_apply_action(static_cast<float>(throttle_cmd));
            float reward = rl_step(dt, done);
            total_reward += reward;
            // Observe next state
            std::vector<float> next_obs = rl_observation();
            double next_alt = static_cast<double>(next_obs[0]) * 10000.0;
            double next_vz  = static_cast<double>(next_obs[1]) * 100.0;
            int next_alt_bin, next_vz_bin;
            discretise(next_alt, next_vz, next_alt_bin, next_vz_bin);
            // Q-learning update
            double best_next = Q[next_alt_bin][next_vz_bin][0];
            for (int a = 1; a < NUM_ACTIONS; ++a) {
                if (Q[next_alt_bin][next_vz_bin][a] > best_next) {
                    best_next = Q[next_alt_bin][next_vz_bin][a];
                }
            }
            Q[alt_bin][vz_bin][action] += alpha * (reward + gamma * best_next - Q[alt_bin][vz_bin][action]);
            // Move to next state
            alt_bin = next_alt_bin;
            vz_bin  = next_vz_bin;
            // Break if episode finished
            if (done) break;
        }
        // Decay ε slightly after each episode
        if (epsilon > 0.01) epsilon *= 0.995;
        // Report training progress
        std::cout << "Episode " << episode + 1 << "/" << max_episodes
                  << " finished with total reward " << total_reward
                  << (done ? " (terminated)" : " (timeout)") << std::endl;
    }

    // Save or print Q-table if needed
    std::cout << "Training complete." << std::endl;
    return 0;
}