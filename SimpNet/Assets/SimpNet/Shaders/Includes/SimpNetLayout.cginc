#ifndef _SIMPNETLAYOUT
#define _SIMPNETLAYOUT

/*
    Forward Propagation
*/

// Layer 1
// x, y : origin
// z, w : width, height
#define txL1Area
#define txL1s                   int4(0, 0, 1024, 32)        // 32x32 x 32
#define txL1a                   int4(0, 0, 1024, 32)        // 32x32 x 32
#define txL1Max                 int4(0, 0, 256, 32)         // 16x16 x 32
#define txL1iMax                int4(0, 0, 256, 32)         // 16x16 x 32
#define txWL1                   int4(0, 0, 27, 32)          // 3x3x3 x 32
#define txBL1                   int4(0, 0, 1, 32)           // 1 x 32

// Layer 2
#define txL2Area
#define txL2s                   int4(0, 0, 196, 64)         // 14x14 x 64
#define txL2a                   int4(0, 0, 196, 64)         // 14x14 x 64
#define txL2Max                 int4(0, 0, 49, 64)          // 7x7 x 64
#define txL2iMax                int4(0, 0, 49, 64)          // 7x7 x 64

// Layer 3
#define txL3Area
#define txL3s                   int4(0, 0, 9, 128)          // 3x3 x 128
#define txL3a                   int4(0, 0, 9, 128)          // 3x3 x 128
#define txL3Max                 int4(0, 0, 1, 128)          // 1 x 128
#define txL3iMax                int4(0, 0, 1, 128)          // 1 x 128

// Layer 4
#define txFC1Area
#define txFC1s                  int4(0, 0, 1, 128)          // 1x128
#define txFC1a                  int4(0, 0, 1, 128)          // 1x128

// Layer 5
#define txFC2Area
#define txFC2s                  int4(0, 0, 1, 128)          // 1x128
#define txFC2a                  int4(0, 0, 1, 128)          // 1x128

// Layer 6
#define txFC3Area
#define txFC3s                  int4(0, 0, 1, 12)           // 1x12
#define txFC3o                  int4(0, 0, 1, 12)           // 1x12

/*
    Back Propagation
*/

// B1
#define txB1                    int4(768, 0, 256, 512)
#define txDBW3Area              int4(142, 256, 1, 12)     // 1x12
#define txDW3Area               int4(128, 256, 12, 128)   // 12x128
#define txDBW2Area              int4(140, 256, 1, 128)    // 1x128
#define txDW2Area               int4(0, 256, 128, 128)    // 128x128
#define txDBW1Area              int4(141, 256, 1, 128)    // 1x128
#define txDW1Area               int4(0, 0, 256, 256)      // 2x2 x 128x128

// B2
#define txB2                    int4(512, 0, 256, 512)
#define txEMax3Area             int4(196, 112, 2, 256)    // 2x2 x 1x128
#define txDB3Area               int4(4, 384, 1, 128)      // 1x128
#define txEConv3Area            int4(0, 0, 4, 512)        // 4x4 x 1x128
#define txDiConv3Area           int4(196, 0, 56, 112)     // 7x7 x 8x16
#define txDKern3Area            int4(4, 0, 192, 384)      // 3x3 x 64x128

// B3
#define txB3                    int4(256, 768, 256, 256)
#define txEMax2Area             int4(96, 112, 56, 56)     // 7x7 x 8x8
#define txDB2Area               int4(208, 0, 1, 64)       // 1x64
#define txEConv2Area            int4(96, 0, 112, 112)     // 14x14 x 8x8
#define txDKern2Area            int4(0, 0, 96, 192)       // 3x3 x 32x64

// B4
#define txB4                    int4(0, 0, 512, 512)
#define txPConv2Area            int4(252, 0, 144, 144)    // 18x18 x 8x8
#define txEMax1Area             int4(396, 0, 64, 128)     // 16x16 x 4x8
#define txDB1Area               int4(460, 24, 1, 32)      // 1x32
#define txEConv1Area            int4(252, 144, 128, 256)  // 32x32 x 4x8
#define txDiConv1Area           int4(0, 0, 252, 504)      // 63x63 x 4x8
#define txDKern1Area            int4(460, 0, 24, 36)      // 3x3x4 x 8x3

// Weight Initialization

#define txInitKern1             int2(24, 384)
#define txInitKern2             int2(192, 256)
#define txInitKern3             int2(0, 0)
#define txInitW1                int2(192, 0)
#define txInitW2                int2(288, 256)
#define txInitW3                int2(0, 384)

#define txInitB1                int2(448, 0)
#define txInitB2                int2(449, 0)
#define txInitB3                int2(450, 0)
#define txInitBw1               int2(451, 0)
#define txInitBw2               int2(452, 0)
#define txInitBw3               int2(453, 0)

// Global Vars

#define eps 0.00001

#define txTimer                 int2(128, 1023)
#define txLCTrain               int2(129, 1023)

/*
    Functions
*/

inline bool insideArea(in int4 area, int2 px)
{
    [flatten]
    if (px.x >= area.x && px.x < (area.x + area.z) &&
        px.y >= area.y && px.y < (area.y + area.w))
    {
        return true;
    }
    return false;
}

inline float LoadValue(in Texture2D<float> tex, in int2 re)
{
    return tex.Load(int3(re, 0));
}

inline float4 LoadValue(in Texture2D<float4> tex, in int2 re)
{
    return tex.Load(int3(re, 0));
}

inline void StoreValue(in int2 txPos, in float value, inout float col,
    in int2 fragPos)
{
    col = all(fragPos == txPos) ? value : col;
}

inline void StoreValue(in int2 txPos, in float4 value, inout float4 col,
    in int2 fragPos)
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