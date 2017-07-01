#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;



/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;//cy---------------------------------------
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.5; //cy: 1.5 before 30--------------------------------

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.57;//cy:0.57 before: 30;----------------------------

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // state dim
  n_x_ = x_.size();

  //augumented state dim (5*state+2*noise)
  n_aug_ = n_x_ + 2;

  // Number of sigma points 
  n_sig_ = 2*n_aug_ + 1;

   //set measurement dimension, radar can measure r, phi, and r_dot
  n_z_ = 3;

  //dimension of matrix with predicted sigma points 
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // spreading parameter:Sigma points distance to mean in relation to error elipse
  lambda_ = 3 - n_aug_; 

  // vector for Weights of sigma points
  weights_ = VectorXd(n_sig_); 

  // Measurement noise covariance matrices initialization
  R_radar_ = MatrixXd(n_z_, n_z_);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0,std_radrd_*std_radrd_;
  R_lidar_ = MatrixXd(2, 2);
  R_lidar_ << std_laspx_*std_laspx_,0,
              0,std_laspy_*std_laspy_;
  // set small value handling
  low = 0.0001; //cy:EPS=0.001
}

UKF::~UKF() {}


// normalize Angles: from -Pi to Pi
void UKF::NormalizeAngle(double x_diff(3)) { //cy:NormAng
    while (x_diff(3)> M_PI) x_diff(3)-= 2.*M_PI;  //cy ang jess:phi uda:x_diff(3)-------------------
    while (x_diff(3)<-M_PI) x_diff(3)+= 2.*M_PI; //cy ang jess:phi uda:x_diff(3)-------------------
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  //-----------------------------------------------------RUNTER
  if (!is_initialized_) {
    // initialize covariance matrix P_
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;
    if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Laser does not measure velocity ->set 0
      x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0;
      // Deal with the special case initialisation problems
      if (fabs(x_(0)) < low and fabs(x_(1)) < low){ //cy: low=EPS-------------------------
      x_(0) = low;
      x_(1) = low;
    } 
      
    else if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      float rho = measurement_pack.raw_measurements_[0]; 
      float phi = measurement_pack.raw_measurements_[1]; 
      float rho_dot = measurement_pack.raw_measurements_[2]; 
      // Coordinates convertion from polar to cartesian
      px = rho * cos(phi); 
      py = rho * sin(phi);
      vx = rho_dot * cos(phi);
      vy = rho_dot * sin(phi);
      float v  = sqrt(vx * vx + vy * vy);
      x_ << px, py, v, 0, 0;
    }

    //set vector for weights for covariance and mean
    double weights_(0) = lambda_/(lambda_+n_aug_);
    for (int i=1; i<2*n_aug_+1; i++) {              //2n+1 weights
        double weights_(i) = 0.5/(n_aug_+lambda_);
    }

    // Save the initiall timestamp for dt calculation
    previous_timestamp_ = measurement_pack.timestamp_; //cy: time_us_------------

    // Done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // Calculate the timestep between measurements in seconds
  double dt = (measurement_pack.timestamp_ - previous_timestamp_);
  dt /= 1000000.0; // dt in seconds: from micro s to s
  previous_timestamp_ = measurement_pack.timestamp_;

  Prediction(dt);

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    //cout << "Radar " << measurement_pack.raw_measurements_[0] << " " << measurement_pack.raw_measurements_[1] << endl;
      UpdateRadar(measurement_pack);
    }
  if (measurement_pack.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    //cout << "Lidar " << measurement_pack.raw_measurements_[0] << " " << measurement_pack.raw_measurements_[1] << endl;
      UpdateLidar(measurement_pack);
  }
}
  



/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
 
  //create augmented mean state
  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_ = MatrixXd(n_aug_, n_sig_);
Xsig_aug_.col(0) = x_aug;

double sqrt_lambda_n_aug = sqrt(lambda_+n_aug_); //only calculate once
VectorXd sqrt_lambda_n_aug_L;

for (int i = 0; i< n_aug_; i++)  {
    Xsig_aug_.col(i+1)        = x_aug + sqrt_lambda_n_aug_L * L.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug - sqrt_lambda_n_aug_L * L.col(i);
}

  //predict sigma points
  for (int i = 0; i< n_sig_; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > low) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;

  }

  // Predicted state mean
  x_ = Xsig_pred_ * weights_;

    // Predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points
    P_ = P_ + weights_(i) * Xsig_pred_.col(i); //------------x_diff(3)???
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // Set measurement dimension---------------------------------kupfer
  int n_z = 2;
  // Create matrix for sigma points in measurement space
  // Transform sigma points into measurement space
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);
  UpdateUKF(meas_package, Zsig, n_z);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
 //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  //define spreading parameter
  double lambda = 3 - n_aug;

  //create matrix with sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_); //n_sig= 2 * n_aug + 1 -----------------------

  // Transform sigma points into measurement space-----------------cy kupfer
  for (int i = 0; i < n_sig_; i++) {
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
    // Measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);          //r
    Zsig(1,i) = atan2(p_y,p_x);                   //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / Zsig(0,i);   //r_dot
  }
  UpdateUKF(meas_package, Zsig, n_z);
}

}

// update function for both sensors
void UKF::UpdateUKF(MeasurementPackage meas_package, MatrixXd Zsig, int n_z){

  //vector for mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred  = Zsig * weights_;

  //create matrix for predicted measurement covariance
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {

    // Residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // Angle normalization
    NormalizeAngle(&(z_diff(1)));
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  // Add measurement noise covariance matrix--------------------------cy kupfer
  MatrixXd R = MatrixXd(n_z, n_z);
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar
    R = R_radar_;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER){ // Lidar
    R = R_lidar_;
  }
  S = S + R;

   //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x, n_z);
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_sig_; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar
      // Angle normalization
      NormalizeAngle(&(z_diff(1)));
    }

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x_;
    // Angle normalization
    NormalizeAngle(&(x_diff(3)));
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Raw Measurements
  VectorXd z = meas_package.raw_measurements_;

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar
    // Angle normalization
    NormalizeAngle(&(z_diff(1)));
  }

  //update state mean and covariance matrix
  x = x + K * z_diff;
  P = P - K*S*K.transpose();

  // Calculate NIS
  double NIS = z.transpose() * S.inverse() * z; //cy:z uda:z_diff
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar
    NIS_radar_ = NIS;
  }
  else {
    NIS_laser_ = NIS;
  }
  
}