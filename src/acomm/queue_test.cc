#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>


/**
 * The purpose of this file is to test using a queue to share data between two
 * threads. The first thread is responsible for putting data onto the queue
 * while the other thread is responsible for taking data off the queue
 */

struct locked_queue {
  std::queue<std::vector<double>> q;
  std::mutex m;

  locked_queue() {
    this->q = std::queue<std::vector<double>>();
  }

  locked_queue(locked_queue&&) = delete;
  locked_queue& operator=(locked_queue&&) = delete;
  
  size_t size() { return this->q.size(); }

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
  this->m.lock();
  std::cout << "enqueue_data: lock acquired\n";
  this->q.push(d);
  std::cout << "data pushed to queue\n";
  this->m.unlock();
  std::cout << "enqueue_data: lock released\n";
  return true;
}

std::vector<double> locked_queue::dequeue_data() {
  this->m.lock();
  std::cout << "dequeue_data: lock acquired\n";
  std::vector<double> res = this->q.front();
  std::cout << "data acquired from queue\n";
  this->q.pop();
  std::cout << "data popped from queue\n";
  this->m.unlock();
  std::cout << "dequeue_data: lock released\n";
  return res;
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

/**
 * Compute average of elements in @p d
 *
 * @param d vector of @p double s
 *
 * @return average of elements of @p d
 */
double average(const std::vector<double>& d) {
  double total(static_cast<double>(d.size()));
  double length(static_cast<double>(d.size()));
  for (const double& x : d) {
    total += x;
  }
  return total / length;
}

/**
 * This function is called by the enqueueing thread and handles the creation of
 * data and enqueueing it to @p locked_queue q
 *
 * @param q @p locked_queue for enqueueing data
 */
void enqueue_data_to_queue(locked_queue& q, int num_data,
    std::atomic<bool>& stop_flag) {

  for (int i = 0; i < num_data; i++) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto data = create_data(20, 0, 10);
    q.enqueue_data(data);
    std::cout << "Added data to queue\n";
  }

  // Tell threads to stop
  stop_flag.store(true, std::memory_order_release);
}

/**
 * This function is called by the dequeuing thread and handles the processing of
 * the data from @p locked_queue. The data is average and printed to stdout.
 *
 * @param q @p locked_queue for dequeueing data
 */
void compute_average_from_queue(locked_queue& q,
    std::atomic<bool>& stop_flag) {

  std::cout << "watching queue!\n";
  while(q.size() == 0)
  while (!stop_flag.load(std::memory_order_release) || q.size() != 0) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "Attempting to dequeue data...\n";
    std::vector<double> data = q.dequeue_data();
    std::cout << "Average: " << average(data) << std::endl;
  }
  std::cout << "queue is empty. Exiting...\n";
}

// ----------------------------------------------------------------------------
//  MAIN
//-----------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  locked_queue q;
  
  std::atomic<bool> stop_flag(false);

  // Create two threads, one for enqueuing and one for dequeueing
  std::thread t1(enqueue_data_to_queue, std::ref(q), 5, std::ref(stop_flag));
  std::thread t2(compute_average_from_queue, std::ref(q), std::ref(stop_flag));
 
  t1.join();
  t2.join(); 
  return 0;
}
