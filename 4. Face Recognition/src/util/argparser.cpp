#include "argparser.h"

#include "log.h"

#include <sstream>
#include <iomanip>

void ArgParser::setArgInfo(const std::vector<Arg>& args)
{
	m_args = args;
	m_longNameToArgIdx.clear();
	m_shortNameToArgIdx.clear();

	for (size_t argIdx = 0; argIdx < m_args.size(); ++argIdx) {
		const Arg& arg = m_args[argIdx];

		auto shortIt = m_shortNameToArgIdx.emplace(arg.shortName, argIdx);
		if (!shortIt.second) {
			logDebug() << "Developer warning: short arg name duplication:" << arg.shortName;
		}

		auto longIt = m_longNameToArgIdx.emplace(arg.longName, argIdx);
		if (!longIt.second) {
			logDebug() << "Developer warning: long arg name duplication:" << arg.longName;
		}
	}
}

std::string ArgParser::generateUsageMessage(const std::string& usageString)
{
	std::stringstream ss;
	ss << "Usage: " << usageString << std::endl;

	if (!m_args.empty())
		ss << "Available options:" << std::endl;
	
	for (Arg arg : m_args) {
		ss << "  " << std::right << std::setw(4) << "-" + arg.shortName << std::left << ", --" << std::setw(30) << arg.longName;
		if (!arg.description.empty()) {
			ss << " " << arg.description;
		}
		ss << std::endl;
	}

	return ss.str();
}

size_t ArgParser::parseArgs(int argc, char* argv[])
{
	if (m_args.empty()) {
		logDebug() << "Won't parse args because arg info is empty";
		return 0;
	}

	size_t paramIdx = 1;
	while (argc) {
		bool fistDash = *argv[0] == '-';
		bool secondDash = fistDash && (*argv)[1] == '-';

		if (!fistDash)
			return paramIdx;

		// For clarity
		bool isLongName = secondDash;

		char* argname = argv[0] + (isLongName ? 2 : 1);

		const std::map<std::string, size_t>& argMap = isLongName ? m_longNameToArgIdx : m_shortNameToArgIdx;

		auto argIt = argMap.find(argname);
		if (argIt == argMap.end()) {
			return paramIdx;
		}

		const Arg& arg = m_args[argIt->second];
		const char* param = nullptr;
		if (arg.hasParam) {
			if (argc <= 1) {
				return paramIdx;
			}
			--argc;
			++argv;
			param = argv[0];
		}

		if (!processArg(arg, param)) {
			return paramIdx;
		}

		--argc;
		++argv;
		++paramIdx;
	}

	return 0;
}
