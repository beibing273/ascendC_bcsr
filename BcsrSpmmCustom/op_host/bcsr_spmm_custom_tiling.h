
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BcsrSpmmCustomTilingData)
  TILING_DATA_FIELD_DEF(int32_t, M);
  TILING_DATA_FIELD_DEF(int32_t, N);
  TILING_DATA_FIELD_DEF(int32_t, K);

  // 行窗口总数
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);

  // 均分行窗口给每个cube core
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);

  // 处理K不对齐
  TILING_DATA_FIELD_DEF(uint32_t, lastKLength);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BcsrSpmmCustom, BcsrSpmmCustomTilingData)
}
