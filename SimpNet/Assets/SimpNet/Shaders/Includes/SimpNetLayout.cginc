#ifndef _SIMPNETLAYOUT
#define _SIMPNETLAYOUT

/*
    Forward Propagation
*/

// Layer 1
// x, y : origin
// z, w : width, height
#define txL1                    int4(0, 512, 256, 256)
#define txKern1Area             int4(193, 0, 24, 36)      // 3x3x3 x 8x4
#define txBias1Area             int4(192, 0, 1, 32)       // 1x32
#define txConv1Area             int4(0, 0, 128, 256)      // 32x32 x 4x8
#define txMax1Area              int4(128, 0, 32, 256)     // 16x16 x 2x16
#define txiMax1Area             int4(160, 0, 32, 256)     // 16x16 x 2x16

// Layer 2
#define txL2                    int4(256, 512, 256, 256)
#define txKern2Area             int4(0, 0, 96, 192)       // 3x3 x 32x64
#define txBias2Area             int4(0, 192, 1, 64)       // 1x64
#define txConv2Area             int4(96, 0, 112, 112)     // 14x14 x 8x8
#define txMax2Area              int4(96, 112, 56, 56)     // 7x7 x 8x8
#define txiMax2Area             int4(152, 112, 56, 56)    // 7x7 x 8x8

// Layer 3
#define txL3                    int4(512, 512, 256, 512)
#define txKern3Area             int4(6, 0, 192, 384)      // 3x3 x 64x128
#define txBias3Area             int4(198, 0, 1, 128)      // 1x128
#define txConv3Area             int4(0, 0, 4, 512)        // 4x4 x 1x128
#define txMax3Area              int4(4, 0, 2, 256)        // 2x2 x 1x128
#define txiMax3Area             int4(4, 256, 2, 256)      // 2x2 x 1x128

// Layer 4
#define txL4                    int4(768, 512, 256, 512)
#define txW1Area                int4(0, 0, 256, 256)      // 2x2 x 128x128
#define txW1BiasArea            int4(0, 256, 1, 128)      // 1x128
#define txFC1s                  int4(1, 256, 1, 128)      // 1x128
#define txFC1a                  int4(2, 256, 1, 128)      // 1x128

// Layer 5
#define txL5                    int4(0, 768, 128, 256)
#define txW2Area                int4(0, 0, 128, 128)      // 128x128
#define txW2BiasArea            int4(0, 128, 1, 128)      // 1x128
#define txFC2s                  int4(1, 128, 1, 128)      // 1x128
#define txFC2a                  int4(2, 128, 1, 128)      // 1x128

// Layer 6
#define txL6                    int4(128, 768, 32, 64)
#define txW3Area                int4(0, 0, 24, 64)        // 12x128 x 2x0.5
#define txW3BiasArea            int4(24, 0, 1, 12)        // 1x12
#define txSoftout1              int4(25, 0, 1, 12)        // 1x12
#define txSoftout2              int4(26, 0, 1, 12)        // 1x12

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
#define txDKern1Area            int4(460, 0, 24, 36)      // 3x3x3 x 8x4

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