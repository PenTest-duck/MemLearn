from zep_cloud.client import Zep

client = Zep(api_key=API_KEY)

# You can choose any user ID, but we recommend using your internal user ID
user_id = "your_internal_user_id"

new_user = client.user.add(
    user_id=user_id,
    email="jane.smith@example.com",
    first_name="Jane",
    last_name="Smith",
)
