#include <iostream>
#include <queue>
#include <vector>
#include <random>


/**
 * The purpose of this file is to test using a queue to share data between two
 * threads. The first thread is responsible for putting data onto the queue
 * while the other thread is responsible for taking data off the queue
 */

/**
 * Creates a @p vector<double> of @p n data points selected randomly on the
 * interval @p [low, high)
 *
 * @param n number of data points created
 * @param low low end of data range (inclusive)
 * @param high high end of the data range (exclusive)
 *
 * @return a @p vector<double> of @p n randomly selected data points on the
 * range @p [low, high)
 */
std::vector<double> create_data(size_t n, int low, int high) {
  std::vector<double> v(n);

  // Create random number generator
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(low, high);

  // Create vector of random data
  for (size_t i = 0; i < n; i++) {
    v[i] = dis(gen);
  }

  return v;
}

/**
 * Adds data @p d to queue @p q
 *
 * @param q a @p std::queue<std::vector<double>> that stores data @p d
 * @param d a @p std::vector<double> of data to be added to queue @p q
 *
 * @return @p true if operation was successful, @p false otherwise
 */
bool enqueue_data(std::queue<std::vector<double>>& q, std::vector<double>& d) {
  return false;
}

/**
 * Takes data from queue @p q
 *
 * @param q @p std::queue<std::vector<double>> type queue containing data
 *
 * @return @p true if successful, @p false otherwise
 */
bool dequeue_data(std::queue<std::vector<double>>&q) {
  return false;
}

// ----------------------------------------------------------------------------
//  MAIN
//-----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  std::cout << "Running async queue tests...\n";
  return 0;
}
