#include "RGBMoCapLiveLinkCommandlet.h"
#include "Async/TaskGraphInterfaces.h"
#include "Containers/Ticker.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformTime.h"
#include "Misc/CoreMisc.h"

URGBMoCapLiveLinkCommandlet::URGBMoCapLiveLinkCommandlet() {
    IsClient = false; IsServer = false; IsEditor = false; LogToConsole = false;
}
int32 URGBMoCapLiveLinkCommandlet::Main(const FString&) {
    double Last = FPlatformTime::Seconds();
    while (!IsEngineExitRequested()) {
        double Now = FPlatformTime::Seconds();
        FTaskGraphInterface::Get().ProcessThreadUntilIdle(ENamedThreads::GameThread);
        FTSTicker::GetCoreTicker().Tick((float)(Now - Last));
        Last = Now;
        FPlatformProcess::Sleep(.005f);
    }
    return 0;
}
