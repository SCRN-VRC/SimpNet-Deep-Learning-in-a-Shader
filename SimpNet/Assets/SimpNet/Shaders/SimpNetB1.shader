Shader "SimpNet/SimpNetB1"
{
    Properties
    {
        _Layer1 ("Layer 6", 2D) = "black" {}
        _FrameBuffer ("Backprop 1 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Backprop 1"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "SimpNetLayout.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            Texture2D<float3> _CamIn;
            Texture2D<float3> _FrameBuffer;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));
                col = _Time.y < 1.0 ? 0..xxx : col;

                [branch]
                if (insideArea(txDW3Area, px))
                {
                    col.r = 1.0;
                }
                else if (insideArea(txDBW3Area, px))
                {
                    col.r = 0.9;
                }
                else if (insideArea(txDW2Area, px))
                {
                    col.r = 0.8;
                }
                else if (insideArea(txDBW2Area, px))
                {
                    col.r = 0.7;
                }
                else if (insideArea(txDW1Area, px))
                {
                    col.r = 0.6;
                }
                else if (insideArea(txDBW1Area, px))
                {
                    col.r = 0.5;
                }

                return col;
            }
            ENDCG
        }
    }
}