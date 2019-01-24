

def train(model, db, dl_train, criterion, optimizer, lr_scheduler, n_epochs=30):
    for epoch in range(n_epochs):
            if not early_stop_flag:
                # -----------------
                #  Train
                # -----------------
                db.train()
                lr_scheduler.step()
                model.train()
                running_loss_train = 0
                for i_batch, sample_batched in tqdm(enumerate(dl_train), desc='Train Epoch {}'.format(epoch + 1),
                                                    total=len(dl_train), unit='batch'):
                    overall_iter = int(i_batch + epoch * len(dl_train))

                    # load data
                    X = sample_batched[0].to(device)
                    y = sample_batched[1].to(device)

                    # zero the gradients
                    optimizer.zero_grad()

                    # forward
                    y_hat = s(model(X))
                    loss = criterion(y_hat, y.view(batch_size, -1))

                    # backward
                    loss.backward()
                    optimizer.step()

                    # statistics
                    running_loss_train += loss.item()

                    if debug and overall_iter % log_period_iter == log_start_iter:
                        writer.add_scalar('loss/train', loss.item() , overall_iter)

                epoch_loss_train = running_loss_train / len(db)
                print('Train Loss: {:.4f}'.format(epoch_loss_train))
                # ---------------------
                #  Validation
                # ---------------------
                db.val()
                model.eval()
                dl_val = DataLoader(db, batch_size=batch_size, num_workers=num_workers, shuffle=True)
                running_loss_val = 0
                for i_batch, sample_batched in tqdm(enumerate(dl_val), desc='Val Epoch {}'.format(epoch + 1),
                                                    total=len(dl_val), unit='batch'):

                    # load data
                    X = sample_batched[0].to(device)
                    y = sample_batched[1].to(device)

                    # forward
                    y_hat = s(model(X))
                    loss = criterion(y_hat, y.view(batch_size, -1))

                    # statistics
                    running_loss_val += loss.item()

                epoch_loss_val = running_loss_val / len(db)
                print('Val Loss: {:.4f}'.format(epoch_loss_val))

                if debug:
                    writer.add_scalar('loss/val', epoch_loss_val, overall_iter)
                    writer.add_image('X', X[0].detach(), overall_iter)
                    # writer.add_image('y', y[0].detach().view(patch_size[0], patch_size[1]))
                    # writer.add_image('y_hat', y_hat[0].detach().view(patch_size[0], patch_size[1]))

                if min_val_loss - epoch_loss_val > 1e-4:
                    if min_val_loss == np.inf:
                        print('Val_loss {}. \nSaving models...'.format(epoch_loss_val))
                    else:
                        print('Val_loss improved by {0:.6f}. \nSaving models...'.format(min_val_loss - epoch_loss_val))
                    torch.save(model.state_dict(), os.path.join(writer_dir, 'model_best.pth'))
                    min_val_loss = epoch_loss_val

                    # save results to csv
                    train_results = pd.DataFrame({'model': model.__class__.__name__,
                                                  'subsample': subsample,
                                                  'patch': patch_size[0],
                                                  'stride': patch_stride[0],
                                                  'batch': batch_size,
                                                  'lr': lr,
                                                  'val_loss': epoch_loss_val,
                                                  'epoch': epoch + 1,
                                                  'run_name': ''.join(run_name)}, index=[0])
                    train_results.to_csv(os.path.join(writer_dir, 'train_results.csv'))
                    early_stop_counter = 0

                else:
                    early_stop_counter += 1

                if debug:
                    writer.add_scalar('Epoch', epoch + 1, overall_iter)

                if early_stop_counter == early_stop:
                    early_stop_flag = True
                    print('\nEarly stopping due to non-decreasing validation loss for {} epochs\n'.format(early_stop))
