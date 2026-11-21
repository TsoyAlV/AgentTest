-- Run against database rag as an administrator. Safe to run again.
BEGIN;
INSERT INTO public.app_roles(name)
VALUES ('employee'),('писатель'),('chatter')
ON CONFLICT(name) DO NOTHING;

INSERT INTO public.app_users(auth_subject)
VALUES ('demo-employee'),('demo-writer'),('demo-chatter')
ON CONFLICT(auth_subject) DO NOTHING;

INSERT INTO public.user_roles(user_id,role_id)
SELECT u.id,r.id
FROM (VALUES
    ('demo-employee','employee'),
    ('demo-writer','писатель'),
    ('demo-chatter','chatter')
) AS assignment(subject,role_name)
JOIN public.app_users u ON u.auth_subject=assignment.subject
JOIN public.app_roles r ON r.name=assignment.role_name
ON CONFLICT DO NOTHING;
COMMIT;

SELECT u.id,u.auth_subject,u.is_active,
       array_agg(r.name ORDER BY r.name) AS roles
FROM public.app_users u
JOIN public.user_roles ur ON ur.user_id=u.id
JOIN public.app_roles r ON r.id=ur.role_id
WHERE u.auth_subject IN ('demo-employee','demo-writer','demo-chatter')
GROUP BY u.id,u.auth_subject,u.is_active
ORDER BY u.id;
