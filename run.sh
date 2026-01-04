#!/bin/bash
set -e
bash install.sh
/root/autodl-tmp/cann-bcsr_test/BcsrSpmmCustomOp/build_out/custom_opp_ubuntu_aarch64.run
cd AclNNInvocation
bash test.sh
cd ..