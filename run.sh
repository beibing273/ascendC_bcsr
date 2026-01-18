#!/bin/bash
set -e
bash install.sh
/root/autodl-tmp/ME_TCF/cann-bcsr/BcsrSpmmCustomOp/build_out/custom_opp_ubuntu_aarch64.run
cd AclNNInvocation
bash test.sh
cd ..