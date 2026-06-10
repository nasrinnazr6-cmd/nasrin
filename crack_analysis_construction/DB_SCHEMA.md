# Database Schema

The active database in this project is `users.db`.

## users

- `id`
- `username`
- `password`
- `role`
- `name`
- `dob`
- `email`
- `contact`
- `address`
- `gender`
- `img`

## prediction

- `id`
- `userid`
- `pred_result`
- `pred_image`
- `orginal_image`
- `avg_depths`
- `confidence`
- `date_time`

## projects

- `id`
- `pcat`
- `pname`
- `ptype`
- `duration`
- `area`
- `client`
- `location`
- `bheight`
- `ftype`
- `fmaterial`
- `pspace`
- `erate`
- `sdesc`
- `poverview`
- `status`
- `img`

## reviews

- `id`
- `project_id`
- `user_id`
- `review_text`
- `date_time`

## engineers

- `id`
- `name`
- `role`
- `experience`
- `speciality`
- `contact`
