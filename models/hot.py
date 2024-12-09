def feature_vis(feats, name): 
    output_shape = (256, 256) 
    # print("feats.size()", feats.size())
    channel_mean = torch.mean(feats, dim=1, keepdim=True) 
    # print("channel_mean1.size()", channel_mean.size())
    channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
    # print("channel_meanF.size()", channel_mean.size())
    # print(type(channel_mean))
    channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().detach().numpy() 
    channel_mean = (((channel_mean - np.min(channel_mean))/(np.max(channel_mean)-np.min(channel_mean)))*255).astype(np.uint8)
    savedir = './'
    if not os.path.exists(savedir + '123'): 
        os.makedirs(savedir + '123') 
    channel_mean_colored = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)

    gray_image = cv2.cvtColor(channel_mean_colored, cv2.COLOR_BGR2GRAY)

    heatmap_colored = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)

    blended_image = cv2.addWeighted(channel_mean_colored, 0.1, heatmap_colored, 0.9, 0)

    cv2.imwrite('123/' + name + '_blended.png', blended_image)