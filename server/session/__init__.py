# importing datetime module for now()
import datetime


class Session(object):

    # - Unique id autoincrement
    autoincrement = 0

    def __init__(self):
        super(Session, self).__init__()

        # - Generate a new id
        self.id = Session.autoincrement + 1
        Session.autoincrement = Session.autoincrement + 1

        # - Message list
        self.messages = []

        # - Session votes
        self.votes = []

    def vote(self, id, vote):

        elem = next((x for x in self.votes if x["id"] == id), None)

        if elem:

            if vote == "none":
                v = elem["vote"]
                self.votes.remove(elem)
                if v == "upvote":
                    return -1
                elif v == "downvote":
                    return 1

            if elem["vote"] == vote:
                return 0
            elif vote == "upvote":
                elem["vote"] = vote
                return 2
            elif vote == "downvote":
                elem["vote"] = vote
                return -2
        else:
            self.votes.append({"id": id, "vote": vote})
            if vote == "upvote":
                return 1
            elif vote == "downvote":
                return -1


    def push_message(self, text, user, payload, audio=""):
        msg = {"message": text, "user": user, "payload": payload, "time": datetime.datetime.now().__str__(), "audio": audio}
        self.messages.append(msg)

    def get_id(self):
        return self.id

    def get_messages(self):
        return self.messages
