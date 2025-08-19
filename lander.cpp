// Mars lander simulator
// Version 1.11
// Mechanical simulation functions
// Gabor Csanyi and Andrew Gee, August 2019

// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation, to make use of it
// for non-commercial purposes, provided that (a) its original authorship
// is acknowledged and (b) no modified versions of the source code are
// published. Restriction (b) is designed to protect the integrity of the
// exercise for future generations of students. The authors would be happy
// to receive any suggested modifications by private correspondence to
// ahg@eng.cam.ac.uk and gc121@eng.cam.ac.uk.

#include "lander.h"

// Add at file scope (top of file, outside any function)
static std::ofstream simlog;
static bool simlog_initialized = false;
static double sim_time = 0.0;

// Predict perigiee (perigee) radius from current r, v (SI units).
static double perigiee_radius(const vector3d& r, const vector3d& v)
{
    static const double mu = GRAVITY * MARS_MASS;   // m^3/s^2

    const double rmag = r.abs();
	const double vmag = v.abs();
    if (rmag <= 0.0) return MARS_RADIUS;            // guard
    
	const double v_exit = sqrt(mu / rmag);  // escape velocity at current radius

    const vector3d hvec = r ^ v;                     // specific angular momentum
    const double   h = hvec.abs();
    if (h < 1e-8 && vmag < v_exit) {
        // Nearly radial: treat current radius as perigiee (degenerate “orbit”)
        double rp = rmag;
        double floor_rp = MARS_RADIUS + 1.0;
        if (rp < floor_rp) rp = floor_rp;
        return 0;
    }

    const double v2 = v.abs2();
    const double eps = 0.5 * v2 - mu / rmag;         // specific energy
    double term = 1.0 + (2.0 * eps * h * h) / (mu * mu);
    if (term < 0.0) term = 0.0;                      // numeric guard
    const double e = sqrt(term);

    const double denom = mu * (1.0 + e);
    double rp = (denom > 0.0) ? (h * h) / denom : rmag;

    double floor_rp = MARS_RADIUS + 1.0;
    if (rp < floor_rp) rp = floor_rp;
    return rp;
}

static void aim_thrust_along(const vector3d& dir_world_unit)
{
    // Map "body +Z in world" to Euler angles (roll=0)
    double pitch = asin(dir_world_unit.y);
    double yaw = atan2(dir_world_unit.x, dir_world_unit.z);
    orientation = vector3d(pitch * 180.0 / M_PI, yaw * 180.0 / M_PI, 0.0);
}


// De-orbit to a target perigiee altitude band using small **retrograde** trims only.
// Returns true while still de-orbiting (caller should return early); false when done.
static bool deorbit_stage(double target_hp, double band_halfwidth)
{
    // Current state
    vector3d r = position;
    vector3d v = velocity;
    const double rmag = r.abs();
    const double vmag = v.abs();
    if (vmag <= 1e-6) return false;   // nothing to aim at

    // Estimate perigee altitude
    const double rp = perigiee_radius(r, v);
    const double hp = rp - MARS_RADIUS;

    // If perigiee already within target band, we're done here
    if (hp >= (target_hp - band_halfwidth) && hp <= (target_hp + band_halfwidth)) {
        throttle = 0.0;
        // Keep a safe retrograde-ish attitude for upcoming aerobrake
        stabilized_attitude = false;
        aim_thrust_along(-v.norm());
        return false;
    }

    // If perigiee is **above** the band => do a gentle **retrograde** burn to lower it
    if (hp > target_hp + band_halfwidth) {
        stabilized_attitude = false;
        aim_thrust_along(-v.norm());  // burn opposite to velocity (true retrograde)
        // Gentle throttle to trim; adjust 0.20–0.40 as needed
        double t = 0.25;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        throttle = t;
        return true;  // still de-orbiting
    }

    // If perigiee is **below** the band, don’t prograde-burn (fuel costly & risky). Just coast.
    throttle = 0.0;
    stabilized_attitude = false;
    aim_thrust_along(-v.norm());
    return false;
}


void autopilot(void)
// Autopilot to adjust the engine throttle, parachute and attitude control
{
	// ---- Orbital Exit and De-orbiting autopilot ----
    {
		// Choose a perigiee target and tolerance band for de-orbiting.
        const double target_hp = 40000.0;  // meters
        const double band_halfwidth = 3000.0;   // meters

        // Only shape the orbit when still high (vacuum-like). Below this, aerobrake instead.
        const double h_atm_start = 120000.0; // meters

        // Current altitude
        const double h = position.abs() - MARS_RADIUS;

        if (h > h_atm_start) {
            if (deorbit_stage(target_hp, band_halfwidth)) {
                // Still de-orbiting this frame
                return;
            }
            // If deorbit_stage returned false, perigiee is set (or we’re coasting) — fall through
        }
 

	// ---- Landing and descent autopilot ----
    
     // Controller parameters 
    const double Kh = 0.01;    //  
    const double Kp = 0.6;     // Controller gain
    const double delta = 0.1;  // Throttle bias to counteract gravity


    // Compute radial unit vector
    vector3d er = position.norm();

    // Actual descent rate (positive downwards)
    double descent_rate = velocity * er;  // Dot product

	// Horizontal velocity component
	vector3d horizontal_velocity = velocity - (descent_rate * er);

	
     // Target descent rate (note the negative sign)
        double v_target = -(0.5 + Kh * h);

        // Error term (positive if descending too fast)
        double error = v_target - descent_rate;

        // Controller output
        double Pout = Kp * error;

        // Adjust throttle using clamped output
        if (Pout <= -delta || h > 10000) {
            throttle = 0.0;
        }
        else if (Pout >= 1.0 - delta) {
            throttle = 1.0;
        }
        else {
            throttle = delta + Pout;
        }

        // Safety check
        if (throttle < 0.0) throttle = 0.0;
        if (throttle > 1.0) throttle = 1.0;

        // Always stabilize attitude (engine points downward)
        stabilized_attitude = true;

        // Extension Tasks
        // Deploy parachute if altitude is above 1000m and descent rate is high
        if (h > 1000.0 && h < 10000 && parachute_status == NOT_DEPLOYED) {
            if (safe_to_deploy_parachute()) {
                parachute_status = DEPLOYED;
            }
        }
        // If the parachute is deployed, ensure the attitude is stabilized
        if (parachute_status == DEPLOYED) {
            stabilized_attitude = true;
        }
        // If the parachute is lost, disable stabilization
        if (parachute_status == LOST) {
            stabilized_attitude = false;
        }
    }
}

static vector3d compute_acceleration(const vector3d& pos, const vector3d& vel) {
    // Gravitational acceleration: a = -G * M / r^2
    vector3d r_vec = pos;
    double r_mag = r_vec.abs();
    vector3d gravity = -GRAVITY * MARS_MASS / (r_mag * r_mag) * r_vec.norm();

    // Thrust (in world frame)
    vector3d thrust = thrust_wrt_world();

    // Drag
    double rho = atmospheric_density(pos);
	double LANDER_AREA = M_PI * LANDER_SIZE * LANDER_SIZE; // Cross-sectional area of the lander
    double CHUTE_AREA = 5.0 * (2.0 * LANDER_SIZE) * (2.0 * LANDER_SIZE); // Cross-sectional area of the parachute


    vector3d drag = -0.5 * rho * LANDER_AREA * vel.abs() * vel * DRAG_COEF_LANDER;

    if (parachute_status == DEPLOYED) {
        drag += -0.5 * rho * CHUTE_AREA * vel.abs() * vel * DRAG_COEF_CHUTE;
    } 

	// Total acceleration
    double total_mass = UNLOADED_LANDER_MASS + FUEL_DENSITY * fuel;
    return (thrust + drag) / total_mass + gravity;
}

void rk4_step(vector3d& position, vector3d& velocity, double dt) {
    // k1 estimates
    vector3d a1 = compute_acceleration(position, velocity);
    vector3d v1 = velocity;

    // k2 estimates
    vector3d v2 = velocity + 0.5 * dt * a1;
    vector3d p2 = position + 0.5 * dt * v1;
    vector3d a2 = compute_acceleration(p2, v2);

    // k3 estimates
    vector3d v3 = velocity + 0.5 * dt * a2;
    vector3d p3 = position + 0.5 * dt * v2;
    vector3d a3 = compute_acceleration(p3, v3);

    // k4 estimates
    vector3d v4 = velocity + dt * a3;
    vector3d p4 = position + dt * v3;
    vector3d a4 = compute_acceleration(p4, v4);

    // Combine for final updates
    position = position + (dt / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4);
    velocity = velocity + (dt / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
}

void verlet_step(void) {
    // Step 1: Compute acceleration from current position and velocity
    vector3d a_t = compute_acceleration(position, velocity);
    // Step 2: Update position
    position = position + velocity * delta_t + 0.5 * a_t * (delta_t * delta_t);
    // Step 3: Compute new acceleration at updated position
    vector3d a_t_new = compute_acceleration(position, velocity);
    // Step 4: Update velocity
    velocity = velocity + 0.5 * (a_t + a_t_new) * delta_t;
}

void euler_step(void) {
    // Step 1: Compute acceleration from current position and velocity
    vector3d a_t = compute_acceleration(position, velocity);
    // Step 2: Update position
    position = position + velocity * delta_t;
    // Step 3: Update velocity
    velocity = velocity + a_t * delta_t;
}

void sim_logging() {
    // Compute altitude and v_target for logging
    double altitude = position.abs() - MARS_RADIUS;
    const double Kh = 0.01;
    double v_target;
    vector3d er = position.norm();

    if (altitude > 10000.0) {
        v_target = velocity * er;
    }
    else {
        v_target = -(0.5 + Kh * altitude);
    }

    // Log data only if file is open
    if (simlog.is_open()) {
        simlog << sim_time << ","
            << position.x << "," << position.y << "," << position.z << ","
            << velocity.x << "," << velocity.y << "," << velocity.z << ","
            << altitude << ","
            << throttle << ","
            << v_target << "\n";
    }

    sim_time += delta_t;
}

void numerical_dynamics (void)
  // This is the function that performs the numerical integration to update the
  // The time step is delta_t (global variable).;
{
    // Initialize log file once
    if (!simlog_initialized) {
        simlog.open("lander_simulation.csv");
        simlog << "time,x,y,z,vx,vy,vz,altitude,throttle,v_target\n"; // CSV header
        simlog_initialized = true;
    }

// runge-kutte integration method
rk4_step(position, velocity, delta_t);


// verlet integration method
// verlet_step();


// euler integration method
// euler_step();

 /*
 // Compute altitude and v_target for logging
 double altitude = position.abs() - MARS_RADIUS;
 const double Kh = 0.01;
 double v_target;
 vector3d er = position.norm();

 if (altitude > 10000.0) {
     v_target = velocity * er;
 }
 else {
     v_target = -(0.5 + Kh * altitude);
 }

 // Log data only if file is open
 if (simlog.is_open()) {
     simlog << sim_time << ","
         << position.x << "," << position.y << "," << position.z << ","
         << velocity.x << "," << velocity.y << "," << velocity.z << ","
         << altitude << ","
         << throttle << ","
         << v_target << "\n";
 }

 sim_time += delta_t;
 */
 // Here we can apply an autopilot to adjust the thrust, parachute and attitude
 if (autopilot_enabled) autopilot();

 // Here we can apply 3-axis stabilization to ensure the base is always pointing downwards
 if (stabilized_attitude) attitude_stabilization();
}

void initialize_simulation (void)
  // Lander pose initialization - selects one of 10 possible scenarios
{
  // The parameters to set are:
  // position - in Cartesian planetary coordinate system (m)
  // velocity - in Cartesian planetary coordinate system (m/s)
  // orientation - in lander coordinate system (xyz Euler angles, degrees)
  // delta_t - the simulation time step
  // boolean state variables - parachute_status, stabilized_attitude, autopilot_enabled
  // scenario_description - a descriptive string for the help screen

  scenario_description[0] = "circular orbit";
  scenario_description[1] = "descent from 10km";
  scenario_description[2] = "elliptical orbit, thrust changes orbital plane";
  scenario_description[3] = "polar launch at escape velocity (but drag prevents escape)";
  scenario_description[4] = "elliptical orbit that clips the atmosphere and decays";
  scenario_description[5] = "descent from 200km";
  scenario_description[6] = "a circular aerostationary orbit";
  scenario_description[7] = "";
  scenario_description[8] = "";
  scenario_description[9] = "";

  switch (scenario) {

  case 0:
    // a circular equatorial orbit
    position = vector3d(1.2*MARS_RADIUS, 0.0, 0.0);
    velocity = vector3d(0.0, -3247.087385863725, 0.0);
    orientation = vector3d(0.0, 90.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 1:
    // a descent from rest at 10km altitude
    position = vector3d(0.0, -(MARS_RADIUS + 10000.0), 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = vector3d(0.0, 0.0, 90.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = true;
    autopilot_enabled = false;
    break;

  case 2:
    // an elliptical polar orbit
    position = vector3d(0.0, 0.0, 1.2*MARS_RADIUS);
    velocity = vector3d(3500.0, 0.0, 0.0);
    orientation = vector3d(0.0, 0.0, 90.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 3:
    // polar surface launch at escape velocity (but drag prevents escape)
    position = vector3d(0.0, 0.0, MARS_RADIUS + LANDER_SIZE/2.0);
    velocity = vector3d(0.0, 0.0, 5027.0);
    orientation = vector3d(0.0, 0.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 4:
    // an elliptical orbit that clips the atmosphere each time round, losing energy
    position = vector3d(0.0, 0.0, MARS_RADIUS + 100000.0);
    velocity = vector3d(4000.0, 0.0, 0.0);
    orientation = vector3d(0.0, 90.0, 0.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = false;
    autopilot_enabled = false;
    break;

  case 5:
    // a descent from rest at the edge of the exosphere
    position = vector3d(0.0, -(MARS_RADIUS + EXOSPHERE), 0.0);
    velocity = vector3d(0.0, 0.0, 0.0);
    orientation = vector3d(0.0, 0.0, 90.0);
    delta_t = 0.1;
    parachute_status = NOT_DEPLOYED;
    stabilized_attitude = true;
    autopilot_enabled = false;
    break;

  case 6:
	  // a circular aerostationary orbit
	  position = vector3d(AEROSTATIONARY_RADIUS, 0.0, 0.0);
      velocity = vector3d(0.0, sqrt(GRAVITY * MARS_MASS / AEROSTATIONARY_RADIUS), 0.0);
      orientation = vector3d(0.0, 0.0, 90.0);
      delta_t = 0.1;
      parachute_status = NOT_DEPLOYED;
      stabilized_attitude = true;
      autopilot_enabled = false;

  case 7:
    break;

  case 8:
    break;

  case 9:
    break;

  }
  // Always (re)initialize logging after scenario selection
  if (simlog.is_open()) {
      simlog.close();
  }
  simlog.open("lander_simulation.csv");
  simlog << "time,x,y,z,vx,vy,vz,altitude,throttle,v_target\n";
  simlog_initialized = true;
  sim_time = 0.0;
}
