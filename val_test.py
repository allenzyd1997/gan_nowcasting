
def validation(model, dataloader, args):
    # Validate the generator performance
    criterion = nn.MSELoss()
    loss_am = AverageMeter()
    p_bar = tqdm(dataloader)
    for i, img in enumerate(p_bar):
        img = img.cuda(non_blocking=True)
        img = torch.squeeze(img, -1)
        # init the valid and the fake matrix here 
        real_imgs = Variable(img.type(Tensor))
        fst_half = real_imgs[:, :args.M, : , : ]
        scd_half = real_imgs[:, args.M:, : , : ]
        z = Variable(Tensor(np.random.normal(0, 1, (1, args.val_batch ,8 ,8 ,8))))
        
        fake_output = model(fst_half, z)

        g_loss = criterion(scd_half,fake_output)
        loss_am.update(g_loss.item())
        infor_per_iter = "VAL Epoch: {iteration}/{iterations:4}. gen_loss: {g_loss:.4f}".format(
                    iteration = i,
                    iterations= len(dataloader),
                    g_loss=loss_am.avg,
                )
        p_bar.set_description(infor_per_iter)
        p_bar.update()
    p_bar.close()
    save(infor_per_iter,args.exp_name, "./loss_result/")
    return loss_am

def test(model, dataloader):
    criterion = nn.MSELoss()
    loss_gen_av = AverageMeter()
    p_bar = tqdm(len(dataloader))
    rec = 0 
    for i, (img, apth) in enumerate(dataloader):
        img = img.cuda(non_blocking=True)
        img = torch.squeeze(img, -1)
        # init the valid and the fake matrix here 
        real_imgs = Variable(img.type(Tensor))
        fst_half = real_imgs[:, :args.M, : , : ]
        scd_half = real_imgs[:, args.M:, : , : ]

        z = Variable(Tensor(np.random.normal(0, 1, (1, args.val_batch ,8 ,8 ,8))))
        
        fake_output = model(fst_half, z)

        g_loss = criterion(scd_half,fake_output)
        loss_gen_av.update(g_loss.item())

        # if i % args.rec_iter == 1:
        if True:
            pth = os.path.join(args.img_test_pth, str(i))
            if not os.path.isdir(pth):
                os.makedirs(pth)
            # 只选择Batch中的第一套图片做存储
            ri = real_imgs[0].cpu().detach().numpy()
            fo = fake_output[0].cpu().detach().numpy()
            for idx, img_it in enumerate(ri):
                # 还原图像
                img_gt = np.uint8(img_it * 80)
                mask = img_gt < 1
                img_gt = 255 * mask + (1 - mask) * img_gt
                # 存储图像
                sv_pth = os.path.join(pth, "real_img_"+str(idx)+'.png')
                cv2.imwrite(sv_pth, img_gt)
            for idx, img_it in enumerate(fo):
                # 还原图像
                img_gt = np.uint8(img_it * 80).reshape(args.H,args.W,1)
                mask = img_gt < 1
                img_gt = 255 * mask + (1 - mask) * img_gt
                # 存储图像
                sv_pth = os.path.join(pth,'fake_out_'+str(idx)+'.png')
                cv2.imwrite(sv_pth, img_gt)

        p_bar.set_description("Test Epoch: {iteration}/{iterations:4}. gen_loss: {g_loss:.4f}".format(
                    iteration = i,
                    iterations= 600,
                    g_loss=loss_gen_av.avg,
                    ) )
        p_bar.update()
    p_bar.close()
