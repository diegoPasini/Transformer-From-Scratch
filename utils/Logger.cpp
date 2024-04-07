#include <iostream>
#include <string>

class Logger {
public:
    static void logRaw(const std::string& message) {
        std::cout << message << std::endl;
    }
    
    static void log(const std::string& message) {
        size_t pos = std::string(__FILE__).find_last_of("/\\");
        if (pos != std::string::npos) {
            std::cout << std::string(__FILE__).substr(pos+1) + ":" + message << std::endl;
        } else {
            std::cout << std::string(__FILE__) + ":" + message << std::endl;
        }
    }

    static void debug(const std::string& message) {
        size_t pos = std::string(__FILE__).find_last_of("/\\");
        if (pos != std::string::npos) {
            std::cout << "DEBUG:" + std::string(__FILE__).substr(pos+1) + ":" + message << std::endl;
        } else {
            std::cout << "DEBUG:" + std::string(__FILE__) + ":" + message << std::endl;
        }
    }

    static void info(const std::string& message) {
        size_t pos = std::string(__FILE__).find_last_of("/\\");
        if (pos != std::string::npos) {
            std::cout << "INFO:" + std::string(__FILE__).substr(pos+1) + ":" + message << std::endl;
        } else {
            std::cout << "INFO:" + std::string(__FILE__) + ":" + message << std::endl;
        }
    }

    static void warning(const std::string& message) {
        size_t pos = std::string(__FILE__).find_last_of("/\\");
        if (pos != std::string::npos) {
            std::cout << "WARNING:" + std::string(__FILE__).substr(pos+1) + ":" + message << std::endl;
        } else {
            std::cout << "WARNING:" + std::string(__FILE__) + ":" + message << std::endl;
        }
    }

    static void error(const std::string& message) {
        size_t pos = std::string(__FILE__).find_last_of("/\\");
        if (pos != std::string::npos) {
            std::cout << "ERROR:" + std::string(__FILE__).substr(pos+1) + ":" + message << std::endl;
        } else {
            std::cout << "ERROR:" + std::string(__FILE__) + ":" + message << std::endl;
        }
    }

private:
    static std::string getFilename(const std::string& filepath) {
        size_t pos = filepath.find_last_of("/\\");
        if (pos != std::string::npos) {
            return filepath.substr(pos + 1);
        } else {
            return filepath;
        }
    }
};
