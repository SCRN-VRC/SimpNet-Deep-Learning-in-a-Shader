#ifndef _SIMPNETLAYOUT
#define _SIMPNETLAYOUT

/*
    Forward Propagation
*/

// Layer 1
// x, y : origin
// z, w : width, height

#define txL1Area                uint4(0, 0, 1024, 96)

#define txL1s                   uint4(0, 0, 1024, 32)        // 32x32 x 32
#define txL1a                   uint4(0, 32, 1024, 32)       // 32x32 x 32
#define txL1Max                 uint4(0, 64, 256, 32)        // 16x16 x 32
#define txL1iMax                uint4(256, 64, 256, 32)      // 16x16 x 32
#define txWL1                   uint4(512, 64, 27, 32)       // 3x3x3 x 32
#define txBL1                   uint4(539, 64, 1, 32)        // 1 x 32

// Layer 2
#define txL2Area                uint4(0, 224, 779, 64)

#define txL2s                   uint4(0, 0, 196, 64)         // 14x14 x 64
#define txL2a                   uint4(196, 0, 196, 64)       // 14x14 x 64
#define txL2Max                 uint4(392, 0, 49, 64)        // 7x7 x 64
#define txL2iMax                uint4(441, 0, 49, 64)        // 7x7 x 64
#define txWL2                   uint4(490, 0, 288, 64)       // 3x3x32 x 64
#define txBL2                   uint4(778, 0, 1, 64)         // 1 x 64

// Layer 3
#define txL3Area                uint4(0, 96, 597, 128)

#define txL3s                   uint4(576, 0, 9, 128)        // 3x3 x 128
#define txL3a                   uint4(585, 0, 9, 128)        // 3x3 x 128
#define txL3Max                 uint4(594, 0, 1, 128)        // 1 x 128
#define txL3iMax                uint4(595, 0, 1, 128)        // 1 x 128
#define txWL3                   uint4(0, 0, 576, 128)        // 3x3x64 x 128
#define txBL3                   uint4(596, 0, 1, 128)        // 1 x 128

// Layer 4
#define txFC1Area               uint4(597, 96, 131, 128)

#define txFC1s                  uint4(128, 0, 1, 128)        // 1 x 128
#define txFC1a                  uint4(129, 0, 1, 128)        // 1 x 128
#define txWFC1                  uint4(0, 0, 128, 128)        // 128 x 128
#define txBFC1                  uint4(130, 0, 1, 128)        // 1 x 128

// Layer 5
#define txFC2Area               uint4(728, 96, 131, 128)

#define txFC2s                  uint4(128, 0, 1, 128)        // 1 x 128
#define txFC2a                  uint4(129, 0, 1, 128)        // 1 x 128
#define txWFC2                  uint4(0, 0, 128, 128)        // 128 x 128
#define txBFC2                  uint4(130, 0, 1, 128)        // 1 x 128

// Layer 6
#define txFC3Area               uint4(859, 96, 131, 12)

#define txFC3s                  uint4(128, 0, 1, 12)         // 1 x 12
#define txFC3o                  uint4(129, 0, 1, 12)         // 1 x 12
#define txWFC3                  uint4(0, 0, 128, 12)         // 128 x 12
#define txBFC3                  uint4(130, 0, 1, 12)         // 1 x 12

/*
    Back Propagation
*/

// B4
#define txB4Area                uint4(613, 572, 260, 268)

#define txDWFC3                 uint4(0, 256, 128, 12)       // 128 x 12
#define txDWFC3_m               uint4(128, 256, 128, 12)     // 128 x 12
#define txDBFC3                 uint4(258, 0, 1, 12)         // 1 x 12
#define txDBFC3_m               uint4(259, 0, 1, 12)         // 1 x 12
#define txDWFC2                 uint4(0, 0, 128, 128)        // 128 x 128
#define txDWFC2_m               uint4(128, 0, 128, 128)      // 128 x 128
#define txDBFC2                 uint4(256, 0, 1, 128)        // 1 x 128
#define txDBFC2_m               uint4(256, 128, 1, 128)      // 1 x 128
#define txDWFC1                 uint4(0, 128, 128, 128)      // 128 x 128
#define txDWFC1_m               uint4(128, 128, 128, 128)    // 128 x 128
#define txDBFC1                 uint4(257, 0, 1, 128)        // 1 x 128
#define txDBFC1_m               uint4(257, 128, 1, 128)      // 1 x 128

// B3
#define txB3Area                uint4(0, 572, 613, 256)

#define txEMax3                 uint4(610, 0, 1, 128)        // 1 x 128
#define txEL3                   uint4(601, 0, 9, 128)        // 3x3 x 128
#define txDiL3                  uint4(576, 0, 25, 128)       // 5x5 x 128
#define txDwL3                  uint4(0, 0, 576, 128)        // 3x3x64 x 128
#define txDwL3_m                uint4(0, 128, 576, 128)      // 3x3x64 x 128
#define txDbL3                  uint4(611, 0, 1, 128)        // 1 x 128
#define txDbL3_m                uint4(612, 0, 1, 128)        // 1 x 128

// B2
#define txB2Area                uint4(0, 828, 565, 128)

#define txPadL3                 uint4(0, 0, 81, 128)         // 9x9 x 128
#define txEL2Max                uint4(369, 64, 49, 64)       // 7x7 x 64
#define txEL2                   uint4(369, 0, 196, 64)       // 14x14 x 64
#define txDwL2                  uint4(81, 0, 288, 64)        // 3x3x32 x 64
#define txDwL2_m                uint4(81, 64, 288, 64)       // 3x3x32 x 64
#define txDbL2                  uint4(418, 64, 1, 64)        // 1 x 64
#define txDbL2_m                uint4(419, 64, 1, 64)        // 1 x 64

// B1
#define txB1Area                uint4(0, 288, 1024, 284)

#define txPadL2                 uint4(504, 32, 324, 64)      // 18x18 x 64
#define txEL1Max                uint4(504, 96, 256, 64)      // 16x16 x 64
#define txEL1                   uint4(0, 0, 1024, 32)        // 32x32 x 64
#define txDiL1                  uint4(0, 32, 504, 252)       // 63x8 x 63x4
#define txDwL1                  uint4(828, 32, 27, 32)       // 3x3x3 x 32
#define txDwL1_m                uint4(855, 32, 27, 32)       // 3x3x3 x 32
#define txDbL1                  uint4(882, 32, 1, 32)        // 1 x 32
#define txDbL1_m                uint4(883, 32, 1, 32)        // 1 x 32

// Weight Initialization

#define txInitKern1             uint2(547, 128)
#define txInitKern2             uint2(258, 128)
#define txInitKern3             uint2(0, 0)
#define txInitW1                uint2(0, 128)
#define txInitW2                uint2(129, 128)
#define txInitW3                uint2(258, 192)

#define txInitB1                uint2(574, 128)
#define txInitB2                uint2(546, 128)
#define txInitB3                uint2(576, 0)
#define txInitBw1               uint2(128, 128)
#define txInitBw2               uint2(257, 128)
#define txInitBw3               uint2(386, 192)

// Global Vars

#define lr                      0.0002 //0.001
#define alpha                   1.0
#define rho                     0.9
#define epsilon                 1e-07

#define LAYERS_CLASSIFY         25
#define LAYERS_TRAIN            48

#define txTimer                 uint2(1023, 1023)
#define txLC                    uint2(1022, 1023)

/*
    Functions
*/

inline bool insideArea(in uint4 area, uint2 px)
{
    [flatten]
    if (px.x >= area.x && px.x < (area.x + area.z) &&
        px.y >= area.y && px.y < (area.y + area.w))
    {
        return true;
    }
    return false;
}

inline float LoadValue(in Texture2D<float> tex, in uint2 re)
{
    return tex.Load(int3(re, 0));
}

inline float4 LoadValue(in Texture2D<float4> tex, in uint2 re)
{
    return tex.Load(int3(re, 0));
}

inline void StoreValue(in uint2 txPos, in float value, inout float col,
    in uint2 fragPos)
{
    col = all(fragPos == txPos) ? value : col;
}

inline void StoreValue(in uint2 txPos, in float4 value, inout float4 col,
    in uint2 fragPos)
{
    col = all(fragPos == txPos) ? value : col;
}

inline float rand(float p)
{
    p = frac(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return (frac(p) - 0.5) * 2.0;
}

#endif