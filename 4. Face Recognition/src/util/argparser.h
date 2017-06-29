#pragma once

#include <string>
#include <vector>
#include <map>

class ArgParser
{
public:
	struct Arg
	{
		size_t id;
		std::string shortName;
		std::string longName;
		std::string description;
		bool hasParam;
	};

	virtual ~ArgParser() {}

	void setArgInfo(const std::vector<Arg>& args);

	std::string generateUsageMessage(const std::string& usageString);
	
	// Returns 0 on success or invalid argument index (starting with 1)
	size_t parseArgs(int argc, char* argv[]);

protected:
	virtual bool processArg(const Arg& arg, const char* param) = 0;

private:
	std::vector<Arg> m_args;
	std::map<std::string, size_t> m_shortNameToArgIdx;
	std::map<std::string, size_t> m_longNameToArgIdx;
};