#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <string>
#include <vector>
#include <map>

#include <fstream>
#include <iostream>
#include <iomanip>
#include "common.h"


class Timer {
public:
    static void Start(const std::string& name) {
        startTimes[name] = std::chrono::high_resolution_clock::now();
    }

    static void Stop(const std::string& name) {
        endTimes[name] = std::chrono::high_resolution_clock::now();
    }

    static void CalculateAndRecordAll() {
        for (const auto& pair : startTimes) {
            const std::string& name = pair.first;
            if (endTimes.count(name)) {
                std::chrono::duration<double, std::milli> duration = endTimes[name] - startTimes[name];
                Record(name, duration.count());
            }
        }
    }

    static void Record(const std::string& name, double time) {
        timings[name].push_back(time);
    }

    static const std::map<std::string, std::vector<double>>& GetTimings() {
        return timings;
    }

    static void Clear() {
        timings.clear();
        startTimes.clear();
        endTimes.clear();
    }

private:
    static std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> startTimes;
    static std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> endTimes;
    static std::map<std::string, std::vector<double>> timings;
};

class Log {
public:
    static void Write(const std::string& category, const std::string& sampleName, const std::map<std::string, std::vector<double>>& timings,const std::string& smode);
};

#endif // TIMER_H