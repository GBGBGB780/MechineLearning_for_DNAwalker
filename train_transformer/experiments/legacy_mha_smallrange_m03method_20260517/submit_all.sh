#!/bin/bash
set -eo pipefail
source /etc/profile
cd "$(dirname "$0")"
mkdir -p logs outputs
jid=$(/opt/pbs/bin/qsub jobs/run_s15_m03method_origknn_1024.pbs)
echo "submitted s15_m03method_origknn_1024: ${jid}"
jid=$(/opt/pbs/bin/qsub jobs/run_s16_m03method_dualunion_2048.pbs)
echo "submitted s16_m03method_dualunion_2048: ${jid}"
jid=$(/opt/pbs/bin/qsub jobs/run_s17_m03method_corridor_2048.pbs)
echo "submitted s17_m03method_corridor_2048: ${jid}"
jid=$(/opt/pbs/bin/qsub jobs/run_s18_m03method_origwin30_352.pbs)
echo "submitted s18_m03method_origwin30_352: ${jid}"
jid=$(/opt/pbs/bin/qsub jobs/run_s19_m03method_origwin35_861.pbs)
echo "submitted s19_m03method_origwin35_861: ${jid}"
