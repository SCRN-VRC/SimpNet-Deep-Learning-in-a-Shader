Shader "SimpNet/SimpNetL3"
{
    Properties
    {
        _Layer2 ("Layer 2", 2D) = "black" {}
        _L3Gradients ("Layer 3 Gradients", 2D) = "black" {}
        _FrameBuffer ("Layer 3 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Layer 3"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float3> _Layer2;
            Texture2D<float3> _L3Gradients;
            Texture2D<float3> _FrameBuffer;
            float4 _Layer2_TexelSize;
            float4 _L3Gradients_TexelSize;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));

                [branch]
                if (insideArea(txKern3Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        // col.r = rand(col.r) * 0.0017361;
                        
                        // Debugging
                        px -= txKern3Area.xy;
                        int i = px.y % 3;
                        int j = px.x % 3;
                        int k = (px.y / 3);
                        int l = (px.x / 3);
                        col.r = (i + j) / float(k + l + 1.0);
                    }
                }
                else if (insideArea(txBias3Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        // col.r = rand(col.r) * 0.5;

                        // Debugging
                        px -= txBias3Area.xy;
                        col.r = (px.y / 128.0) - 0.5;
                    }
                }
                else if (insideArea(txConv3Area, px))
                {
                    px -= txConv3Area.xy;
                    int i = px.y % 4;
                    int j = px.x % 4;
                    int k = px.y / 4;
                    int i0 = i * 2, i1 = i0 - 1, i2 = i0 + 1;
                    int j0 = j * 2, j1 = j0 - 1, j2 = j0 + 1;

                    // Padding
                    bool bi1 = i1 < 0, bj1 = j1 < 0, bi2 = i2 > 6, bj2 = j2 > 6;
                    bool b02 = bi1 || bj1, b03 = bi1 || bj2, b04 = bi2 || bj1, b05 = bi2 || bj2;

                    float sum = 0.0;
                    [loop]
                    for (int l = 0; l < 64; l++) {
                        sum +=
                            (b02 ? 0.0 : getMax2(_Layer2, int3(j1, i1, l)) * getKern3(_FrameBuffer, int4(0, 0, l, k))) +
                            (bi1 ? 0.0 : getMax2(_Layer2, int3(j0, i1, l)) * getKern3(_FrameBuffer, int4(0, 1, l, k))) +
                            (b03 ? 0.0 : getMax2(_Layer2, int3(j2, i1, l)) * getKern3(_FrameBuffer, int4(0, 2, l, k))) +
                            (bj1 ? 0.0 : getMax2(_Layer2, int3(j1, i0, l)) * getKern3(_FrameBuffer, int4(1, 0, l, k))) +
                            getMax2(_Layer2, int3(j0, i0, l)) * getKern3(_FrameBuffer, int4(1, 1, l, k)) +
                            (bj2 ? 0.0 : getMax2(_Layer2, int3(j2, i0, l)) * getKern3(_FrameBuffer, int4(1, 2, l, k))) +
                            (b04 ? 0.0 : getMax2(_Layer2, int3(j1, i2, l)) * getKern3(_FrameBuffer, int4(2, 0, l, k))) +
                            (bi2 ? 0.0 : getMax2(_Layer2, int3(j0, i2, l)) * getKern3(_FrameBuffer, int4(2, 1, l, k))) +
                            (b05 ? 0.0 : getMax2(_Layer2, int3(j2, i2, l)) * getKern3(_FrameBuffer, int4(2, 2, l, k)));
                    }
                    sum += _FrameBuffer.Load(int3(txBias3Area.xy + int2(0, k), 0));
                    col.r = actFn(sum);
                }
                else if (insideArea(txMax3Area, px))
                {
                    px -= txMax3Area.xy;
                    int i = px.y % 2;
                    int j = px.x % 2;
                    int k = px.x / 2 + (px.y / 2);
                    int i0 = i * 2, i1 = i0 + 1;
                    int j0 = j * 2, j1 = j0 + 1;

                    float m = getConv3(_FrameBuffer, int3(j0, i0, k));
                    m = max(m, getConv3(_FrameBuffer, int3(j0, i1, k)));
                    m = max(m, getConv3(_FrameBuffer, int3(j1, i0, k)));
                    m = max(m, getConv3(_FrameBuffer, int3(j1, i1, k)));
                    col.r = m;
                }
                else if (insideArea(txiMax3Area, px))
                {
                    px -= txiMax3Area.xy;
                    int i = px.y % 2;
                    int j = px.x % 2;
                    int k = px.x / 2 + (px.y / 2);
                    int i0 = i * 2, i1 = i0 + 1;
                    int j0 = j * 2, j1 = j0 + 1;

                    float bu;
                    float m = getConv3(_FrameBuffer, int3(j0, i0, k));
                    col.r = i0 * 4 + j0;
                    
                    m = max(m, bu = getConv3(_FrameBuffer, int3(j0, i1, k)));
                    col.r = (m == bu) ? (i1 * 4 + j0) : col.r;
                    
                    m = max(m, bu = getConv3(_FrameBuffer, int3(j1, i0, k)));
                    col.r = (m == bu) ? (i0 * 4 + j1) : col.r;
                    
                    m = max(m, bu = getConv3(_FrameBuffer, int3(j1, i1, k)));
                    col.r = (m == bu) ? (i1 * 4 + j1) : col.r;

                    if (i == 1 && j == 0 && k == 127) {
                        buffer[0] = float4(col.r, m, 0, 0);
                    }
                }
                return col;
            }
            ENDCG
        }
    }
}