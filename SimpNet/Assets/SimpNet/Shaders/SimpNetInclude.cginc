#ifndef _SIMPNET_INC
#define _SIMPNET_INC

// x, y : origin
// z, w : width, height
#define txKern1Area             int4(0, 0, 3, 288)
#define txBias1Area             int4(0, 288, 1, 32)
#define txConv1Area             int4(3, 0, 64, 512)
#define txMax1Area              int4(67, 0, 16, 512)
#define txiMax1Area             int4(83, 0, 16, 512)

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