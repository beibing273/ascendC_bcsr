#!/bin/bash
set -e
bash install.sh
/root/AcendCFile/cann-bcsr/BcsrSpmmCustomOp/build_out/custom_opp_ubuntu_aarch64.run
cd AclNNInvocation
bash test_reorder_colcondense.sh
cd ..