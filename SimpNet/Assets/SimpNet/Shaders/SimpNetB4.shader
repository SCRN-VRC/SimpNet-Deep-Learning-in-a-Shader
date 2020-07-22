Shader "SimpNet/SimpNetB4"
{
    Properties
    {
        _BackProp3 ("Backprop 3", 2D) = "black" {}
        _FrameBuffer ("Backprop 4 Buffer", 2D) = "black" {}
    }
    SubShader
    {
        
        Pass
        {
            Name "SimpNet Backprop 4"

            CGPROGRAM
            #include "UnityCustomRenderTexture.cginc"
            #include "Includes/SimpNetLayout.cginc"
            #include "Includes/SimpNetFuncs.cginc"
            #pragma vertex CustomRenderTextureVertexShader
            #pragma fragment pixel_shader
            #pragma target 5.0

            RWStructuredBuffer<float4> buffer : register(u1);
            Texture2D<float3> _BackProp3;
            Texture2D<float3> _FrameBuffer;
            float4 _FrameBuffer_TexelSize;

            float3 pixel_shader (v2f_customrendertexture IN) : SV_TARGET
            {
                int2 px = _FrameBuffer_TexelSize.zw * IN.globalTexcoord.xy;
                float3 col = _FrameBuffer.Load(int3(px, 0));

                [branch]
                if (insideArea(txPConv2Area, px))
                {
                    px -= txPConv2Area.xy;
                    col.r = 1.0;
                }
                else if (insideArea(txEMax1Area, px))
                {
                    col.r = 0.9;
                }
                else if (insideArea(txDB1Area, px))
                {
                    col.r = 0.8;
                }
                else if (insideArea(txEConv1Area, px))
                {
                    col.r = 0.7;
                }
                else if (insideArea(txDiConv1Area, px))
                {
                    col.r = 0.6;
                }
                else if (insideArea(txDKern1Area, px))
                {
                    col.r = 0.5;
                }

                return col;
            }
            ENDCG
        }
    }
}