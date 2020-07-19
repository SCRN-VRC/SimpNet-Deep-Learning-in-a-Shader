#ifndef _SIMPNET_INC
#define _SIMPNET_INC

/*
    Forward Propagation
*/

// Layer 1
// x, y : origin
// z, w : width, height
#define txKern1Area             int4(0, 0, 3, 288)
#define txBias1Area             int4(0, 288, 1, 32)
#define txConv1Area             int4(3, 0, 64, 512)
#define txMax1Area              int4(67, 0, 16, 512)
#define txiMax1Area             int4(83, 0, 16, 512)

// Layer 2
#define txKern2Area             int4(99, 0, 96, 192)
#define txBias2Area             int4(195, 0, 1, 64)
#define txConv2Area             int4(99, 192, 112, 112)
#define txMax2Area              int4(211, 0, 7, 448)
#define txiMax2Area             int4(218, 0, 7, 448)

// Layer 3
#define txKern3Area             int4(225, 0, 192, 384)
#define txBias3Area             int4(225, 384, 1, 128)
#define txConv3Area             int4(417, 0, 4, 512)
#define txMax3Area              int4(421, 0, 2, 256)
#define txiMax3Area             int4(423, 0, 2, 256)

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