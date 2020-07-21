Shader "SimpNet/SimpNetL1"
{
    Properties
    {
        _CamIn ("Camera Input", 2D) = "black" {}
        _L1Gradients ("Layer 1 Gradients", 2D) = "black" {}
        _FrameBuffer ("Layer 1 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Layer 1"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float3> _CamIn;
            Texture2D<float3> _L1Gradients;
            Texture2D<float3> _FrameBuffer;
            float4 _CamIn_TexelSize;
            float4 _L1Gradients_TexelSize;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));

                [branch]
                if (insideArea(txKern1Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        // col.r = rand(col.r) * 0.037;
                        
                        // Debugging
                        px -= txKern1Area.xy;
                        int i = px.y % 3;
                        int j = px.x % 3;
                        int k = (px.y / 3) % 3;
                        int l = (px.x / 3) + (px.y / 9) * 8;
                        col.r = i * j * k / (l + 1.0);
                    }
                }
                else if (insideArea(txBias1Area, px))
                {
                    if (_Time.y <= 1.0)
                    {
                        // col.r = px.y * _FrameBuffer_TexelSize.z + px.x;
                        // col.r = rand(col.r) * 0.5;

                        // Debugging
                        px -= txBias1Area.xy;
                        int k = px.y % 32;
                        col.r = px.y / 32.0 - 0.5;
                    }
                }
                else if (insideArea(txConv1Area, px))
                {
                    px -= txConv1Area.xy;

                    int i = px.y % 32;
                    int j = px.x % 32;
                    int k = px.x / 32 + (px.y / 32) * 4;
                    int i0 = i * 2, i1 = i0 + 1, i2 = i0 + 2;
                    int j0 = j * 2, j1 = j0 + 1, j2 = j0 + 2;

                    float sum = 0.0;
                    [unroll]
                    for (int l = 0; l < 3; l++) {
                        sum += _CamIn.Load(int3(j0, i0, 0))[l] * getKern1(_FrameBuffer, int4(0, 0, l, k));
                        sum += _CamIn.Load(int3(j0, i1, 0))[l] * getKern1(_FrameBuffer, int4(0, 1, l, k));
                        sum += _CamIn.Load(int3(j0, i2, 0))[l] * getKern1(_FrameBuffer, int4(0, 2, l, k));
                        sum += _CamIn.Load(int3(j1, i0, 0))[l] * getKern1(_FrameBuffer, int4(1, 0, l, k));
                        sum += _CamIn.Load(int3(j1, i1, 0))[l] * getKern1(_FrameBuffer, int4(1, 1, l, k));
                        sum += _CamIn.Load(int3(j1, i2, 0))[l] * getKern1(_FrameBuffer, int4(1, 2, l, k));
                        sum += _CamIn.Load(int3(j2, i0, 0))[l] * getKern1(_FrameBuffer, int4(2, 0, l, k));
                        sum += _CamIn.Load(int3(j2, i1, 0))[l] * getKern1(_FrameBuffer, int4(2, 1, l, k));
                        sum += _CamIn.Load(int3(j2, i2, 0))[l] * getKern1(_FrameBuffer, int4(2, 2, l, k));
                    }
                    
                    sum += _FrameBuffer.Load(int3(txBias1Area.xy + int2(0, k), 0));
                    col.r = actFn(sum);
                }
                else if (insideArea(txMax1Area, px))
                {
                    px -= txMax1Area.xy;

                    int i = px.y % 16;
                    int j = px.x % 16;
                    int k = px.x / 16 + (px.y / 16) * 2;
                    int i0 = i * 2, i1 = i0 + 1;
                    int j0 = j * 2, j1 = j0 + 1;

                    float m = getConv1(_FrameBuffer, int3(j0, i0, k));
                    m = max(m, getConv1(_FrameBuffer, int3(j0, i1, k)));
                    m = max(m, getConv1(_FrameBuffer, int3(j1, i0, k)));
                    m = max(m, getConv1(_FrameBuffer, int3(j1, i1, k)));
                    col.r = m;
                }
                else if (insideArea(txiMax1Area, px))
                {
                    px -= txiMax1Area.xy;

                    int i = px.y % 16;
                    int j = px.x % 16;
                    int k = px.x / 16 + (px.y / 16) * 2;
                    int i0 = i * 2, i1 = i0 + 1;
                    int j0 = j * 2, j1 = j0 + 1;

                    float bu;
                    float m = getConv1(_FrameBuffer, int3(j0, i0, k));
                    col.r = i0 * 32 + j0;
                    
                    m = max(m, bu = getConv1(_FrameBuffer, int3(j0, i1, k)));
                    col.r = (m == bu) ? (i0 * 32 + j1) : col.r;
                    
                    m = max(m, bu = getConv1(_FrameBuffer, int3(j1, i0, k)));
                    col.r = (m == bu) ? (i1 * 32 + j0) : col.r;
                    
                    m = max(m, getConv1(_FrameBuffer, int3(j1, i1, k)));
                    col.r = (m == bu) ? (i1 * 32 + j1) : col.r;
                }
                return col;
            }
            ENDCG
        }
    }
}