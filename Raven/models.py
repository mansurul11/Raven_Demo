from django.db import models

# Create your models here.
class Category(models.Model):

    #name = models.CharField(max_length=128, unique=True)
    fb1 = models.BooleanField()
    fb2 = models.BooleanField()
    fb3 = models.BooleanField()
    fb4 = models.BooleanField()
    fb5 = models.BooleanField()
    fb6 = models.BooleanField()
    def __unicode__(self):  #For Python 2, use __str__ on Python 3
        return self.fb1, self.fb2, self.fb3,self.fb4, self.fb5, self.fb6

class Blank(models.Model):
    title = models.NOT_PROVIDED
    
    def __unicode__(self):      #For Python 2, use __str__ on Python 3
        return self.title
    
class Pattern(models.Model):
    
    name = models.CharField(max_length=1024)
    pindex = models.IntegerField(default=0)

    def __unicode__(self):      #For Python 2, use __str__ on Python 3
        return self.name

