#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <vector>


/**
 * The purpose of this file is to test using a queue to share data between two
 * threads. The first thread is responsible for putting data onto the queue
 * while the other thread is responsible for taking data off the queue
 */

struct locked_queue {
  std::queue<std::vector<double>> q;
  std::mutex m;
  size_t size;

  locked_queue(size_t n) : size(n) {
    this->q = std::queue<std::vector<double>>();
  }

  /**
   * Adds data @p d to queue @p q
   *
   * @param q a @p std::queue<std::vector<double>> that stores data @p d
   * @param d a @p std::vector<double> of data to be added to queue @p q
   *
   * @return @p true if operation was successful, @p false otherwise
   */
  bool enqueue_data(std::vector<double>& d);

  /**
   * Takes data from queue @p q
   *
   * @param q @p std::queue<std::vector<double>> type queue containing data
   *
   * @return @p std::vector<double> of data popped from queue
   */
  std::vector<double> dequeue_data();
};

bool locked_queue::enqueue_data(std::vector<double>& d) {
  std::lock_guard<std::mutex> log(this->m); 
  this->q.push(d);
  return true;
}

std::vector<double> locked_queue::dequeue_data() {
  std::lock_guard<std::mutex> lock(this->m);
  return this->q.pop();
}


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


// ----------------------------------------------------------------------------
//  MAIN
//-----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  locked_queue q(10);
  return 0;
}
