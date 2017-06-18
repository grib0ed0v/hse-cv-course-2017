#pragma once

#include <ostream>

#define ENABLE_LOGGING 1

#if(ENABLE_LOGGING)
#define logInfo() Logger(Logger::LogLevel::Info, "Info")
#define logWarning() Logger(Logger::LogLevel::Warning, "Warning")
#define logError() Logger(Logger::LogLevel::Error, "Error")
#define logDebug() Logger(Logger::LogLevel::Debug, "DEBUG", __FILE__, __LINE__)
#else
#define logInfo() NoLogger()
#define logWarning() NoLogger()
#define logError() NoLogger()
#define logDebug() NoLogger()

class NoLogger
{
public:
	template<typename T>
	NoLogger& operator<<(const T& val) { return *this; }
};
#endif

class Logger
{
public:
	enum LogLevel
	{
		None,
		Error,
		Warning,
		Info,
		Debug
	};

	Logger(LogLevel level, const char* prefix = nullptr, const char* file = nullptr, int line = 0)
	{
		m_logLevel = level; 
		if (m_logLevel <= s_logLevel) {
			if (prefix)
				s_logStream << prefix;
			if (file)
				s_logStream << '(' << file;
			if (line) {
				s_logStream << ':';
				s_logStream << line;
			}
			if (file)
				s_logStream << ')';
			if (prefix)
				s_logStream << ": ";
		}

	}

	~Logger()
	{
		if (m_logLevel <= s_logLevel) {
			s_logStream << std::endl;
		}
	}

	static void setLogLevel(LogLevel level) { s_logLevel = level; }
	static void setOstream(std::streambuf* buf) { s_logStream.rdbuf(buf); }
	template<typename T>
	Logger& operator<<(const T& val)
	{ 
		if (m_logLevel <= s_logLevel) {
			s_logStream << val << ' ';
		}
		return *this; 
	}

private:
	LogLevel m_logLevel = LogLevel::Info;
	static LogLevel s_logLevel;
	static std::ostream s_logStream;
};

