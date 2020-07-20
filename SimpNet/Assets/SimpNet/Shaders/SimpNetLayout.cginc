#ifndef _SIMPNET_INC
#define _SIMPNET_INC

/*
    Forward Propagation
*/

// Layer 1
// x, y : origin
// z, w : width, height
#define txKern1Area             int4(193, 0, 36, 24)      // 3x3x3 x 4x8
#define txBias1Area             int4(192, 0, 1, 32)       // 1x32
#define txConv1Area             int4(0, 0, 128, 256)      // 32x32 x 4x8
#define txMax1Area              int4(128, 0, 32, 256)     // 16x16 x 2x16
#define txiMax1Area             int4(160, 0, 32, 256)     // 16x16 x 2x16

// Layer 2
#define txKern2Area             int4(0, 0, 96, 192)       // 3x3 x 32x64
#define txBias2Area             int4(0, 192, 1, 64)       // 1x32
#define txConv2Area             int4(96, 0, 112, 112)     // 14x14 x 8x8
#define txMax2Area              int4(96, 112, 56, 56)     // 7x7 x 8x8
#define txiMax2Area             int4(152, 112, 56, 56)    // 7x7 x 8x8

// Layer 3
#define txKern3Area             int4(0, 0, 192, 384)
#define txBias3Area             int4(0, 0, 1, 128)
#define txConv3Area             int4(0, 0, 4, 512)
#define txMax3Area              int4(0, 0, 2, 256)
#define txiMax3Area             int4(0, 0, 2, 256)

// Layer 4
#define txW1Area                int4(425, 0, 256, 256)
#define txW1BiasArea            int4(421, 256, 1, 128)
#define txFC1s                  int4(422, 256, 1, 128)
#define txFC1a                  int4(423, 256, 1, 128)

// Layer 5
#define txW2Area                int4(226, 384, 128, 128)
#define txW2BiasArea            int4(354, 384, 1, 128)
#define txFC2s                  int4(355, 384, 1, 128)
#define txFC2a                  int4(356, 384, 1, 128)

// Layer 6
#define txW3Area                int4(357, 384, 12, 128)
#define txW3BiasArea            int4(369, 384, 1, 12)
#define txSoftout1              int4(370, 384, 1, 12)
#define txSoftout2              int4(371, 384, 1, 12)

/*
    Back Propagation
*/


inline bool insideArea (in int4 area, int2 px)
{
    [flatten]
    if (px.x >= area.x && px.x < (area.x + area.z) &&
        px.y >= area.y && px.y < (area.y + area.w))
    {
        return true;
    }
    return false;
}

inline float LoadValue (in Texture2D<float> tex, in int2 re)
{
    return tex.Load(int3(re, 0));
}

inline void StoreValue (in int2 txPos, in float value, inout float col,
    in int2 fragPos)
{
    col = all(fragPos == txPos) ? value : col;
}

#endif