// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <iterator>
#include <string>
#include <unordered_map>
#include "comm.h"
#include "nccl.h"
#include "colltrace/CollTrace.h"

std::string serializeMap(std::unordered_map<std::string, std::string> map) {
  std::string final_string = "{";
  for (auto& it : map) {
    final_string += it.first;
    final_string += ": ";
    final_string += it.second;
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string = final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "}";
  return final_string;
}

std::string serializeList(std::list<std::string> strings) {
  std::string final_string = "[";
  for (auto& it : strings) {
    final_string += it;
    final_string += ", ";
  }
  if (final_string.size() > 1) {
    final_string = final_string.substr(0, final_string.size() - std::string(", ").size());
  }
  final_string += "]";
  return final_string;
}

std::unordered_map<std::string, std::string> retrieveNCCLInfoMap(ncclInfo info) {
  std::unordered_map<std::string, std::string> infoMap;
  infoMap["opName"] = std::string{info.opName};
  infoMap["count"] = std::to_string(info.count);
  infoMap["datatype"] = std::to_string(info.datatype);
  infoMap["algorithm"] = std::to_string(info.algorithm);
  infoMap["protocol"] = std::to_string(info.protocol);
  infoMap["pattern"] = std::to_string(static_cast<uint8_t>(info.pattern));
  infoMap["channelId"] = std::to_string(info.channelId);
  return infoMap;
}

// Todo: rewrite the two functions below to make them less janky
std::string serializeResults(std::list<ResultInfo>& results, int64_t maxResultPrint=10) {
  auto endIter = results.rbegin();
  std::list<std::string> serializedResults;
  if (maxResultPrint > results.size()) {
    endIter = results.rend();
  } else {
    std::advance(endIter, maxResultPrint);
  }

  for (auto it = results.rbegin(); it != endIter; it++) {
    auto infoMap = retrieveNCCLInfoMap(it->info);
    infoMap["opCount"] = std::to_string(it->opCount);
    // Insert in the reverse order
    serializedResults.emplace_front(serializeMap(infoMap));
  }

  return serializeList(serializedResults);
}

std::string serializeEvents(std::queue<std::unique_ptr<EventInfo>>& events) {
  std::list<std::string> serializedEvents;
  while (!events.empty()) {
    auto eventInfo = std::move(events.front());
    events.pop();
    auto infoMap = retrieveNCCLInfoMap(eventInfo->info);
    infoMap["opCount"] = std::to_string(eventInfo->opCount);
    serializedEvents.emplace_back(serializeMap(infoMap));
  }
  return serializeList(serializedEvents);
}

std::string serializeRing(int* userRanks, int nRank) {
  std::list <std::string> ringList;
  for (int i = 0; i < nRank; i++) {
    ringList.emplace_back(std::to_string(userRanks[i]));
  }
  return serializeList(ringList);
}

std::string serializeRings(ncclComm_t comm) {
  std::list<std::string> serializedRings;
  for (int i = 0; i < comm->nChannels; i++) {
    serializedRings.emplace_back(serializeRing(comm->channels[i].ring.userRanks, comm->nRanks));
  }
  return serializeList(serializedRings);
}

__attribute__((visibility("default"))) ncclResult_t ncclCommDump(
    ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  map["commHash"] = std::to_string(comm->commHash);
  if (comm->collTrace != nullptr) {
    auto traceDump = comm->collTrace->dumpTrace();
    INFO(
      NCCL_ALL,
      "CommDump: Dumping %zu past results, %zu pending events, there %s a current event",
      traceDump.pastResults.size(),
      traceDump.pendingEvents.size(),
      traceDump.currentEvent == nullptr ? "is not" : "is");

    map["CT_pastResults"] = serializeResults(traceDump.pastResults);
    map["CT_pendingEvents"] = serializeEvents(traceDump.pendingEvents);
    if (traceDump.currentEvent != nullptr && traceDump.currentEventState == EventState::RUNNING) {
      auto infoMap = retrieveNCCLInfoMap(traceDump.currentEvent->info);
      infoMap["opCount"] = std::to_string(traceDump.currentEvent->opCount);
      map["CT_currentEvent"] = serializeMap(infoMap);
      auto algorithm = traceDump.currentEvent->info.algorithm;
      auto channelId = traceDump.currentEvent->info.channelId;
      if (algorithm == NCCL_ALGO_RING) {
        map["CT_currentRing"]= serializeRing(comm->channels[channelId].ring.userRanks, comm->nRanks);
      }
    }
  } else {
    INFO(NCCL_ALL, "CommDump: No trace to dump");
  }
  return ncclSuccess;
}
