#include "log.h"

#include <iostream>

#if DEBUG 
	Logger::LogLevel Logger::s_logLevel = Logger::LogLevel::Debug;
#else
	Logger::LogLevel Logger::s_logLevel = Logger::LogLevel::Info;
#endif

std::ostream Logger::s_logStream(std::cout.rdbuf());
