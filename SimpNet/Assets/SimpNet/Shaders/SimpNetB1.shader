Shader "SimpNet/SimpNetB1"
{
    Properties
    {
        _TargetClass ("Target Class #", Int) = 0
        _Layer6 ("Layer 6", 2D) = "black" {}
        _Layer5 ("Layer 5", 2D) = "black" {}
        _Layer4 ("Layer 4", 2D) = "black" {}
        _FrameBuffer ("Backprop 1 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Backprop 1"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float3> _Layer6;
            Texture2D<float3> _Layer5;
            Texture2D<float3> _Layer4;
            Texture2D<float3> _FrameBuffer;
            float4 _FrameBuffer_TexelSize;
            int _TargetClass;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));

                [branch]
                if (insideArea(txDBW3Area, px))
                {
                    px -= txDBW3Area.xy;
                    int i = px.y;
                    col.r = _Layer6.Load(int3(txSoftout2.xy + int2(0, i), 0)).x - (i == _TargetClass ? 1.0 : 0.0);
                }
                else if (insideArea(txDW3Area, px))
                {
                    px -= txDW3Area.xy;
                    int i = px.y;
                    int j = px.x;
                    col.r = _FrameBuffer.Load(int3(txDBW3Area.xy + int2(0, j), 0)).x *
                        _Layer5.Load(int3(txFC2a.xy + int2(0, i), 0)).x;
                }
                else if (insideArea(txDBW2Area, px))
                {
                    px -= txDW2Area.xy;
                    int i = px.y;

                    float sum = 0.0;
                    for (int k = 0; k < 12; k++) {
                        sum += _FrameBuffer.Load(int3(txDBW3Area.xy + int2(0, k), 0)).x *
                            getW3(_Layer6, int2(k, i));
                    }
                    col.r = sum;
                }
                else if (insideArea(txDW2Area, px))
                {
                    px -= txDW2Area.xy;
                    int i = px.y;
                    int j = px.x;
                    col.r = _FrameBuffer.Load(int3(txDBW2Area.xy + int2(0, i), 0)).x *
                        dactFn(_Layer5.Load(int3(txFC2s.xy + int2(0, i), 0)).x) *
                        _Layer4.Load(int3(txFC1a.xy + int2(0, j), 0)).x;

                    if (i == 0 && j == 78)
                    {
                        buffer[0] = float4(col.r * 100, 0, 0, 0);
                    }
                }
                else if (insideArea(txDBW1Area, px))
                {
                    col.r = 0.6;
                }
                else if (insideArea(txDW1Area, px))
                {
                    col.r = 0.5;
                }

                return col;
            }
            ENDCG
        }
    }
}