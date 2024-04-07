#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

class Timer {
private:
    std::chrono::steady_clock::time_point start_time;
    std::vector<std::chrono::milliseconds> logged_times;
    bool running = false;

public:
    void start() {
        if (!running) {
            start_time = std::chrono::steady_clock::now();
            running = true;
        }
    }

    void stop() {
        if (running) {
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            logged_times.push_back(duration);
            running = false;
        }
    }

    void printLoggedTimes() const {
        std::cout << "Logged times (" << logged_times.size() << "):" << std::endl;
        for (const auto& time : logged_times) {
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time);
            auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(time - seconds);
            std::cout << seconds.count() << "." << milliseconds.count() << " seconds" << std::endl;
        }
    }

    void printLastTime() const {
        if (!logged_times.empty()) {
            const auto& lastTime = logged_times.back();
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(lastTime);
            auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(lastTime - seconds);
            std::cout << "Last logged time: " << seconds.count() << "." << milliseconds.count() << std::endl;
        } else {
            std::cout << "No times logged yet." << std::endl;
        }
    }
};
