#include "timer.h"
#include<ctime>


std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> Timer::startTimes;
std::map<std::string, std::chrono::time_point<std::chrono::high_resolution_clock>> Timer::endTimes;
std::map<std::string, std::vector<double>> Timer::timings;

void Log::Write(const std::string& category, const std::string& sampleName, const std::map<std::string, std::vector<double>>& timings,const std::string& smode) {
    std::string filePath = "../output_all/" + category +"_"+ smode+ ".txt";
    std::ofstream outFile(filePath, std::ios::app);
    time_t time_handle=time(nullptr);
    std::tm* local_time=localtime(&time_handle);
    if (!outFile.is_open()) {
        ERROR_LOG("Failed to open log file: %s", filePath.c_str());
        return;
    }
    outFile << "Sample: " << sampleName<<"_"<< smode << std::endl;
    for (const auto& pair : timings) {
        outFile << "  " << pair.first << ":" << std::endl;
        for (size_t i = 0; i < pair.second.size(); ++i) {
            outFile << "    Run " << i + 1 << ": " << std::fixed << std::setprecision(6) << pair.second[i] << " ms" << std::endl;
        }
    }
    outFile<<"Local datetime at test execution: "<<(local_time->tm_year+1900)<<":"<<(local_time->tm_mon+1)<<":"<<local_time->tm_mday
            <<" "<<local_time->tm_hour<<":"<<local_time->tm_min<<":"<<local_time->tm_sec<<std::endl;
    outFile << "----------------------------------------" << std::endl;
    outFile.close();
}
